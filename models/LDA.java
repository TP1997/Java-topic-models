package models;

import java.io.*;
import java.lang.reflect.Type;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.ArrayList;
import java.util.Set;

import java.nio.file.Files;
import java.nio.file.Paths;

import utility.FuncUtils;
import utility.FileUtils;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

public class LDA 
{
	public double alpha; //Hyper-parameter alpha
	public double beta; //Hyper-parameter beta
	public int K; //Number of topics
	
	public int topWords; //Number of most probable words for each topic
	
	public List<List<Integer>> corpus; // Word ID-based corpus
	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus
	
	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary: word -> ID
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary: ID -> word
	public int V; // Vocabulary size
	
	int z[][]; //topic assignments for each word.
	
	int[][] N; //N[d][k] = Number of words in document d assigned to topic k (Omega)
	int[] Nsum; //Nsum[d] = Sum over k of N[d][k]'s, i.e., total number of words in document d
	int[][] M; //M[v][k]=Number of times term v has been assigned to topic k
	int[] Msum; //Msum[k] = Sum over v of M[v][k]'s, i.e., total number of words assigned to topic k
	
	double[][] thetasum; //Cumulative statistics of theta
	double[][] phisum; //Cumulative statistics of phi
	
	int numstats; //size of statistics
	
	private static int THIN_INTERVAL = 20; //sampling lag (?)
	private static int BURN_IN = 100; //burn-in period
	private static int ITERATIONS = 1000; //max iterations
	private static int dispcol = 0;
	private static int SAMPLE_LAG; //sample lag (if -1 only one sample taken)
	
	public double[] multiPros; //Double array used to sample a topic
	
	public String folderPath; // directory containing the corpus
	public String corpusPath; // Path to the topic modeling corpus
	
	public String expName = "LDAmodel";
	
	public double initTime = 0; // Initialization time
	public double iterTime = 0;
	
	public LDA(String datapath, int inNumTopics, double inAlpha, double inBeta,
			int inNumIterations, int inTopWords)throws Exception
	{
		this(datapath, inNumTopics, inAlpha, inBeta, inNumIterations, inTopWords, "LDAmodel");
	}
	
	public LDA(String datapath, int inNumTopics, double inAlpha, double inBeta, 
			int inNumIterations, int inTopWords, String inExpName)
			throws Exception
	{
		alpha = inAlpha;
		beta = inBeta;
		K = inNumTopics;
		ITERATIONS = inNumIterations;
		topWords = inTopWords;
		
		expName = inExpName;
		corpusPath = datapath;
		folderPath = "results/";
		File dir = new File(folderPath);
		if (!dir.exists())
			dir.mkdir();
		
		System.out.println("Reading topic modeling corpus: " + datapath);
		
		// Load vocabulary
		word2IdVocabulary = FileUtils.loadToken2idx(datapath+"token2idx.json");
		id2WordVocabulary = FileUtils.loadIdx2token(datapath+"idx2token.json");
		// Load corpus
		corpus = FileUtils.loadCorpus(datapath+"train_vidx.txt");
		numDocuments = 20000;
		numWordsInCorpus = 178980;
		
		V = word2IdVocabulary.size();
		
		// Initialize count variables.
		N = new int[numDocuments][K];
		Nsum = new int[numDocuments];
		M = new int[V][K];
		Msum = new int[K];
		
		multiPros = new double[K];
		for(int i=0; i<K; i++) {
			multiPros[i] = 1.0 / K;
		}
		
		z = new int[numDocuments][];
		
		System.out.println("Corpus size: "+numDocuments+" docs, "+numWordsInCorpus+" words");
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of common topics: " + K);
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("Number of sampling iterations: " + ITERATIONS);
		System.out.println("Number of top topical words: " + topWords);
		
		initialize();
	}
	
	//Randomly initialize topic assignments
	public void initialize()throws IOException
	{
		System.out.println("Randomly initializing topic assignments ...");
		
		long startTime = System.currentTimeMillis();
		for (int i=0; i<numDocuments; i++) {
			List<Integer> topics = new ArrayList<Integer>();
			int docSize = corpus.get(i).size();
			z[i] = new int[docSize];
			for (int j=0; j<docSize; j++) {
				int z_ij = FuncUtils.nextDiscrete(multiPros); // Sample a topic
				// Increase counts
				N[i][z_ij] += 1;
				Nsum[i] += 1;
				M[corpus.get(i).get(j)][z_ij] += 1;
				Msum[z_ij] += 1;
				z[i][j] = z_ij;
			}
		}
		initTime =System.currentTimeMillis()-startTime;
	}
	
	public void inference()throws IOException
	{
		//writeDictionary();
		
		System.out.println("Running Gibbs sampling inference: ");
		
		// init sampler statistics
		if (SAMPLE_LAG>0) {
			thetasum = new double[numDocuments][K];
			phisum = new double[K][V];
			numstats = 0;
		}
		
		System.out.println("Sampling "+ITERATIONS+" iterations with burn-in of "+ BURN_IN +" (B/S="
				+THIN_INTERVAL+").");
		
		long startTime = System.currentTimeMillis();
		
		for (int i=0; i<ITERATIONS; i++) {
			// For every z_i=z_mn
			for(int m=0; m<z.length; m++) {
				for(int n=0; n<z[m].length; n++) {
					// Sample from p(z_i|Z_-i,W)
					int z_mn = sampleFullConditional(m,n);
					z[m][n] = z_mn; 
				}
			}
			
			if((i<BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.print("Burn-in");
				dispcol++;
			}
			// Display progress
			if((i>BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.print("Sampling");
				dispcol++;
			}
			// Get statistics after burn-in
			if((i>BURN_IN) && (SAMPLE_LAG>0) && (i%SAMPLE_LAG==0)) {
				updateParams();
				System.out.print("|");
				if(i%THIN_INTERVAL != 0)
					dispcol++;
			}
			if(dispcol>=100) {
				System.out.println();
				dispcol = 0;
			}
		}
		
		iterTime = System.currentTimeMillis()-startTime;
		//System.out.println("Writing output from the last sample ...");
		//write();
		System.out.println("Sampling completed!");
	}
	
	/**
	 * Add to the statistics the values of theta and phi for the current state.
	 */
	private void updateParams() {
		for(int m=0; m<numDocuments; m++) {
			for(int k=0; k<K; k++) {
				thetasum[m][k] = (N[m][k]+alpha)/(Nsum[m]+K*alpha);
			}
		}
		for(int k=0; k<K; k++) {
			for(int v=0; v<V; v++) {
				phisum[k][v] = (M[v][k]+beta)/(Msum[k]+V*beta);
			}
		}
		numstats++;
	}
	
	/**
	 * Sample a topic z_i=z_mn from the full conditional distribution p(z_i|z_-i,W)
	 * @param m = document
	 * @param n = word
	 * @return
	 */
	private int sampleFullConditional(int m, int n) {
		// Remove z_i from the count variables
		int z_i = z[m][n]; 
		N[m][z_i] -= 1;
		//Nsum[m] -= 1;
		M[corpus.get(m).get(n)][z_i] -= 1;
		Msum[z_i] -= 1;
		
		// Do multinomial sampling via cumulative method:
		double[] p = new double[K]; // Unnormalized probabilities
		for(int k=0; k<K; k++) {
			p[k] = (M[corpus.get(m).get(n)][k]+beta)/(Msum[k]+V*beta) * (N[m][k]+alpha); 
		}
		// Cumulate multinomial probabilities
		for(int k=1; k<p.length; k++) {
			p[k] += p[k-1];
		}
		System.out.println("Cumulated sum: "+p[p.length-1]);
		// Scaled sample because of unnormalized p's
		double u = Math.random() * p[K-1];
		for(z_i=0; z_i<p.length; z_i++) {
			if(u<p[z_i])
				break;
		}
		
		// Add newly sampled z_i to count variables
		N[m][z_i] += 1;
		M[corpus.get(m).get(n)][z_i] += 1;
		Msum[z_i] += 1;
		
		return z_i;
	}
	
	public void writeTopTopicalWords()throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath+expName+".topWords"));
		for(int tIndex=0; tIndex<K; tIndex++) {
			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for(int wIndex=0; wIndex<V; wIndex++) {
				wordCount.put(wIndex, M[wIndex][tIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);
			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for(Integer index : mostLikelyWords) {
				if(count<topWords) {
					//double pro = (M[index][tIndex]+beta)/(Msum[tIndex]+V*beta);
					//pro = Math.round(pro * 1000000.0) / 1000000.0;
					writer.write(id2WordVocabulary.get(index) + " ");
					count += 1;
				}
				else {
					writer.write("\n");
					break;
				}
			}
		}
		writer.close();
	}
	
	public void write()throws IOException
	{
		writeTopTopicalWords();
	}
	
	public static void main(String[] args) throws Exception
	{
		// public LDA(String pathToCorpus, int inNumTopics, double inAlpha, double inBeta, 
		// int inNumIterations, int inTopWords, String inExpName)
		String datapath = "/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/All_Beauty/20k/";
		LDA lda = new LDA(datapath, 10, 0.1, 0.1, 1000, 10, "All_Beauty-LDA");
		lda.inference();
	}

}
