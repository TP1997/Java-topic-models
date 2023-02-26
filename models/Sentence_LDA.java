package models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import utility.FileUtils;
import utility.FuncUtils;

public class Sentence_LDA {
	public double alpha;
	public double beta;
	public int K;
	
	public List<List<List<Integer>>> corpus; // Word ID-based corpus (d,s,w)
	public int D; // Number of documents
	public HashMap<String, Integer> token2id; // Vocabulary: word -> ID
	public HashMap<Integer, String> id2token; // Vocabulary: ID -> word
	public int V; // Vocabulary size
	ArrayList<List<List<int[]>>> vidxCountTable; // word_id:count representation of corpus
	
	int z[][]; //z[d][s]=Topic assignments of sentence s of document d.
	
	int[][] N; // N[d][k]=Number of sentences in document d assigned to topic k
	int[][] M; // M[k][v]=Number of times topic k has been assigned to term v
	int[] Msum; // Msum[k] = Sum over v of M[v][k]'s, i.e., total number of words assigned to topic k
	
	private static int THIN_INTERVAL = 10; //sampling lag (?)
	private static int BURN_IN = 100; //burn-in period
	private static int ITERATIONS = 1000; //max iterations
	private static int dispcol = 0;
	private static int SAMPLE_LAG = 10; //sample lag (if -1 only one sample taken)
	
	public Sentence_LDA(String dataname, HashMap<String, String> dataConf, int K, double alpha, double beta)
	{
		this.alpha = alpha;
		this.beta = beta;
		this.K = K;
		
		System.out.println("Reading topic modeling corpus...");
		// Load vocabulary
		token2id = FileUtils.loadToken2idx(dataConf.get("token2idx_file"));
		id2token = FileUtils.loadIdx2token(dataConf.get("idx2token_file"));
		// Load corpus
		corpus = new ArrayList<List<List<Integer>>>(); // doc, sen, word
		corpus=FileUtils.loadCorpusSentences(dataConf.get("train_file_vidx"));
		D = Integer.parseInt(dataConf.get("size"));
		V = token2id.size();
		// Load word_id:count representation of corpus
		vidxCountTable = FileUtils.loadCountTable(dataConf.get("train_file_vidx_count"));
		
		// Define count variables
		N = new int[D][K];
		M = new int[K][V];
		Msum = new int[K];
		// Define hidden variables
		z = new int[D][];
		
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of documents: " + D);
		System.out.println("Number of common topics: " + K);
		
		initialize();
	}
	
	//Randomly initialize topic assignments (z's) and update count matrices accordingly
	public void initialize()
	{
		System.out.println("Randomly initializing hidden variables ...");
		
		double[] p = new double[K];
		Arrays.fill(p, 1.0/K);
		for(int d=0; d<D; d++) {
			int Nd = corpus.get(d).size();
			z[d] = new int[Nd];
			for(int s=0; s<Nd; s++) {
				int z_ds = FuncUtils.nextDiscrete(p);
				// Increase counts
				N[d][z_ds] += 1;
				for(int n=0; n<corpus.get(d).get(s).size(); n++) {
					M[z_ds][corpus.get(d).get(s).get(n)] += 1;
				}
				Msum[z_ds] += corpus.get(d).get(s).size();
				z[d][s] = z_ds;
			}
		}
		System.out.println("Initialization done!"); 
	}
	
	public void inference(int n_iter)
	{
		ITERATIONS = n_iter;
		System.out.println("Running Gibbs sampling inference: ");
		System.out.println("Sampling "+ITERATIONS+" iterations with burn-in of "+ BURN_IN +" (B/S="
				+THIN_INTERVAL+").");
		
		for(int i=0; i<ITERATIONS; i++) {
			for(int d=0; d<D; d++) {
				for(int s=0; s<z[d].length; s++) {
					//int z_ds = sampleFullConditional(d,s);
					int z_ds = sampleFullConditional2(d,s);
					z[d][s] = z_ds;
				}
			}
			if((i<BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.println("Burn-in");
			}
			if((i>BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.println("Sampling:"+i+"/"+ITERATIONS);
			}
			// Get statistics after burn-in
			if((i>BURN_IN) && (SAMPLE_LAG>0) && (i%SAMPLE_LAG==0)) {
				updateParams();
			}
		}
		System.out.println("Sampling completed!");
	}
	
	private int sampleFullConditional(int d, int s)
	{
		// Update count variables, i.e., remove effect of z_i
		N[d][z[d][s]] -= 1;
		HashMap<Integer, Integer> nHist = FuncUtils.createCountTable(corpus.get(d).get(s)); // Construct n(v,w_i)
		int nSigma = corpus.get(d).get(s).size();
		for(int v : nHist.keySet()) {
			//System.out.print(v);
			M[z[d][s]][v] -= nHist.get(v);
		}
		Msum[z[d][s]] -= corpus.get(d).get(s).size();
		//for(int n=0; n<corpus.get(d).get(s).size(); n++) {
			//M[z[d][s]][corpus.get(d).get(s).get(n)] -= 1;
		//}
		//Msum[z[d][s]] -= corpus.get(d).get(s).size();
		
		// Construct probability table for GS
		double[] p = new double[K];
		for(int k=0; k<K; k++) {
			double num = 1;
			for(int v : nHist.keySet()) {
				for(int j=1; j<=nHist.get(v); j++) {
					num *= (M[k][v]+beta+nHist.get(v)-j);
				}
			}
			double denom = 1;
			for(int j=1; j<=nSigma; j++) {
				denom *= (Msum[k]+V*beta+nSigma-j);
			}
			p[k] = (N[d][k]+alpha) * num/denom;
		}
		
		int z_i = FuncUtils.nextDiscrete(p);
		// Add newly sampled z_i to count variables
		N[d][z_i] += 1;
		for(int n=0; n<corpus.get(d).get(s).size(); n++) {
			M[z_i][corpus.get(d).get(s).get(n)] += 1;
		}
		Msum[z_i] += corpus.get(d).get(s).size();
		
		return z_i;
	}
	
	private int sampleFullConditional2(int d, int s)
	{
		// Update count variables, i.e., remove effect of z_i
		N[d][z[d][s]] -= 1;
		for(int[] vidx_count : vidxCountTable.get(d).get(s)) { // vidx_count[0]=v, vidx_count[1]=n(v,w_i)
			M[z[d][s]][vidx_count[0]] -= vidx_count[1];
		}
		Msum[z[d][s]] -= corpus.get(d).get(s).size();
		
		// Construct probability table for GS up to constant
		int nSigma = corpus.get(d).get(s).size();
		double[] p = new double[K];
		for(int k=0; k<K; k++) {
			double num = 1;
			for(int[] vidx_count : this.vidxCountTable.get(d).get(s)) {
				for(int j=1; j<=vidx_count[1]; j++) {
					num *= (M[k][vidx_count[0]]+beta+vidx_count[1]-j);
				}
			}
			double denom = 1;
			for(int j=1; j<=nSigma; j++) {
				denom *= (Msum[k]+V*beta+nSigma-j);
			}
			p[k] = (N[d][k]+alpha) * num/denom;
		}
		
		int z_i = FuncUtils.nextDiscrete(p);
		// Add the effect of newly sampled z_i to count variables
		N[d][z_i] += 1;
		for(int n=0; n<corpus.get(d).get(s).size(); n++) {
			M[z_i][corpus.get(d).get(s).get(n)] += 1;
		}
		Msum[z_i] += corpus.get(d).get(s).size();
		return z_i;
	}
	
	private void updateParams() 
	{
		
	}
	
	public static void main(String[] args)
	{
		// Load data_conf.json
		String dataname = "All_Beauty_sentences";
		String datadir = "20k";
		String datapath = "/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/"
				+ dataname+"/"+datadir+"/";
		HashMap<String, String> dataConf = FileUtils.loadDataConf(datapath+"data_conf.json");
		
		// Define model parameters
		double alpha = 0.1;
		double beta = 0.01;
		int K = 20;
		
		Sentence_LDA senLDA = new Sentence_LDA(dataname, dataConf, K, alpha, beta);
		senLDA.inference(1000);
		/*
		int d=0;
		int s=2;
		int e=1;
		ArrayList<List<List<int[]>>> countTable = FileUtils.loadCountTable(datapath+"train_vidx_counts.txt");
		
		List<int[]> list = countTable.get(d).get(s);
		for(int[] entry : list) {
			System.out.println(entry[0]+":"+entry[1]);
		}
		*/
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
