package models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IntSummaryStatistics;
import java.util.List;

import utility.FileUtils;
import utility.FuncUtils;

public class C_LDA_simple {
	public double[][] alpha;
	public double beta;
	public int C; //Number of collections
	public int K; //Number of common topics
	public int[] K_C; // Numbers of private topics in each collection
	
	public String[] dataNames;
	public List<List<List<Integer>>> corpus; // Word ID-based corpus
	public int[] D_C; // Number of documents in each collection
	
	public HashMap<String, Integer> token2id; // Vocabulary: word -> ID
	public HashMap<Integer, String> id2token; // Vocabulary: ID -> word
	public int V; // Vocabulary size (combined)
	
	int z[][][]; //z[c][d][n] = topic assignment of word w_cdn
	
	int[][][] N; // (N[c][d][k]) N[c][k][d] = Number of words in document d of collection c assigned to topic k∈{1...K...K_c}
	int[][] Nsum; // Nsum[c][d] = Sum over N[c][d][k]'s, i.e., total number of words in document d of collection c
	int[][] MI; // M[k][v] = Number of times common topic k∈{1...K} has been assigned to term v
	int[] MIsum; // Sum over MI[k][v]'s, i.e., total number of words assigned to common topic k.
	int[][][]MS; // M[c][k][v] = Number of times private topic k∈K_c/{K} has been assigned to term v
	int[][]MSsum; // MSsum[c][k] = Sum over MS[c][k][v]'s, i.e., total number of words assigned to private topic k of collection c.
	//int[][][]M; // M[c][k][v] = Number of times (common or private) topic k has been assigned to term v in collection c
	
	private static int THIN_INTERVAL = 10; //sampling lag (?)
	private static int BURN_IN = 100; //burn-in period
	private static int ITERATIONS = 1000; //max iterations
	private static int dispcol = 0;
	private static int SAMPLE_LAG = 10; //sample lag (if -1 only one sample taken)
	
	public double initTime = 0; // Initialization time
	public double iterTime = 0; // Gibbs sampling time
	
	public C_LDA_simple(String[] dataNames, HashMap<String, String> dataConf,
			int C, int K, int[] K_C, double alpha, double beta, int numIterations)//throws Exception
	{
		//this.alpha = alpha;
		this.beta = beta;
		this.C = C;
		this.K = K;
		this.K_C = K_C;
		initAlpha(alpha);
		ITERATIONS = numIterations;
		this.dataNames = dataNames;
		
		System.out.println("Reading topic modeling corpus...");
		// Load vocabulary
		token2id = FileUtils.loadToken2idx(dataConf.get("token2idx_file"));
		id2token = FileUtils.loadIdx2token(dataConf.get("idx2token_file"));
		// Load corpus
		corpus = new ArrayList<List<List<Integer>>>();
		for(int c=0; c<C; c++) {
			String key = dataNames[c]+"_train_file_vidx";
			List<List<Integer>> arr = FileUtils.loadCorpus(dataConf.get(key));
			corpus.add(arr);
		}
		// Loading dataset sizes
		D_C = new int[C];
		for(int c=0; c<C; c++) {
			String key = dataNames[c]+"_size";
			D_C[c] = Integer.parseInt(dataConf.get(key)); 
		}
		// Set vocabulary size
		V = token2id.size();
		
		// Initialize count variables
		N = new int[C][][];
		//for(int c=0; c<C; c++) { N[c] = new int[D_C[c]][K+K_C[c]]; }
		for(int c=0; c<C; c++) { N[c] = new int[K+K_C[c]][D_C[c]]; }
		Nsum = new int[C][];
		for(int c=0; c<C; c++) { Nsum[c] = new int[D_C[c]]; }
		MI = new int[K][V];
		MIsum = new int[K];
		MS = new int[C][][];
		for(int c=0; c<C; c++) { MS[c] = new int[K_C[c]][V]; }
		MSsum = new int[C][];
		for(int c=0; c<C; c++) { MSsum[c] = new int[K_C[c]]; }
		
		// Initialize hidden variables
		z = new int[C][][];
		
		//System.out.println("Corpus size: "++" docs, "++" words");
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of common topics: " + K);
		for(int c=0; c<C; c++) {
			System.out.println(dataNames[c]+": Number of private topics: " + K_C[c]);
		}
		System.out.println("Number of sampling iterations: " + ITERATIONS);
		
		initialize();
	}
	
	public void initAlpha(double a)
	{
		alpha = new double[C][];
		for(int c=0; c<C; c++) { 
			alpha[c] = new double[K+K_C[c]];
			Arrays.fill(alpha[c], 1.0*a/(K+K_C[c]));
		}
	}
	
	//Randomly initialize topic assignments (z's)
	public void initialize()
	{
		System.out.println("Randomly initializing hidden variables ...");
		double[][] p = new double[C][];
		for(int c=0; c<C; c++) {
			p[c] = new double[K+K_C[c]];
			for(int k=0; k<K+K_C[c]; k++) {
				p[c][k] = 1.0/(K+K_C[c]);
			}
		}
		
		long startTime = System.currentTimeMillis();
		for(int c=0; c<C; c++) {
			z[c] = new int[D_C[c]][];
			for(int d=0; d<D_C[c]; d++) {
				int docSize = corpus.get(c).get(d).size();
				z[c][d] = new int[docSize];
				for(int n=0; n<docSize; n++) {
					int z_cdn = FuncUtils.nextDiscrete(p[c]);
					// Increase counts
					//N[c][d][z_cdn] += 1;
					N[c][z_cdn][d] += 1;
					Nsum[c][d] += 1;
					if(z_cdn < K) { // common topic
						MI[z_cdn][corpus.get(c).get(d).get(n)] += 1; // !
						MIsum[z_cdn] += 1;
					}
					else {
						MS[c][z_cdn-K][corpus.get(c).get(d).get(n)] += 1;
						MSsum[c][z_cdn-K] += 1;
					}
					z[c][d][n] = z_cdn;
				}
			}
		}
		initTime =System.currentTimeMillis()-startTime;
		System.out.println("Done!");
	}
	
	public void inference()//throws IOException
	{
		System.out.println("Running Gibbs sampling inference: ");
		System.out.println("Sampling "+ITERATIONS+" iterations with burn-in of "+ BURN_IN +" (B/S="
				+THIN_INTERVAL+").");
		
		int minDocs = Arrays.stream(D_C).min().getAsInt(); 
		for(int i=0; i<ITERATIONS; i++) {
			// For every r_i=r_cdn and z_i=z_cdn
			// Do sampling in following order: z[0][0][.],z[1][0][.], z[0][1][.],z[1][1][.], z[0][2][.],z[1][2][.], ...
			for(int d=0; d<minDocs; d++) {
				for(int c=0; c<C; c++) {
					for(int n=0; n<z[c][d].length; n++) {
						int z_i = sampleFullConditional(c, d, n); //!
						z[c][d][n] = z_i;
					}
				}
			}
			// Finish sampling for collections c where D_C[c]>minDocs
			for(int c=0; c<C; c++) {
				for(int d=minDocs; d<D_C[c]; d++) {
					for(int n=0; n<z[c][d].length; n++) {
						int z_i = sampleFullConditional(c, d, n);
						z[c][d][n] = z_i;
					}
				}
			}
			
			if((i<BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.println("Burn-in");
				dispcol++;
			}
			if((i>BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.println("Sampling");
				dispcol++;
				System.out.println("Optimizing alpha...");
				for(int c=0; c<C; c++) {
					updateAlpha(c);
				}
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
		System.out.println("Sampling completed!");
	}
	
	private int sampleFullConditional(int c, int d, int n)
	{
		// Update count variables
		//N[c][d][z[c][d][n]] -= 1;
		N[c][z[c][d][n]][d] -= 1;
		Nsum[c][d] -= 1;
		if(z[c][d][n]<K) { // common topic
			MI[z[c][d][n]][corpus.get(c).get(d).get(n)] -= 1;
			MIsum[z[c][d][n]] -= 1;
		}
		else { // Private topic
			MS[c][z[c][d][n]-K][corpus.get(c).get(d).get(n)] -= 1;
			MSsum[c][z[c][d][n]-K] -= 1;
		}
		
		// construct probability table for GS
		double[] p = new double[K+K_C[c]];
		for(int k=0; k<K; k++) { // First half, common topic probs.
			//p[k] = (N[c][d][k]+alpha[c][k])/(Nsum[c][d]+alphasum[c]) *
				//   (MI[k][corpus.get(c).get(d).get(n)]+beta)/(MIsum[k]+V*beta);
			//p[k] = (N[c][d][k]+alpha[c][k])*(MI[k][corpus.get(c).get(d).get(n)]+beta)/(MIsum[k]+V*beta);
			p[k] = (N[c][k][d]+alpha[c][k])*(MI[k][corpus.get(c).get(d).get(n)]+beta)/(MIsum[k]+V*beta);
		}
		for(int k=0; k<K_C[c]; k++) { // Second half, private topic probs.
			//p[k] = (N[c][d][k]+alpha[c][k])/(Nsum[c][d]+alphasum[c]) *
				//	(MS[c][k][corpus.get(c).get(d).get(n)]+beta)/(MSsum[c][k]+V*beta);
			//p[K+k] = (N[c][d][k]+alpha[c][k])*(MS[c][k][corpus.get(c).get(d).get(n)]+beta)/(MSsum[c][k]+V*beta);
			p[K+k] = (N[c][k][d]+alpha[c][k])*(MS[c][k][corpus.get(c).get(d).get(n)]+beta)/(MSsum[c][k]+V*beta);
		}
		
		/*
		// Cumulate probabilities
		for(int k=1; k<p.length; k++) {
			p[k] += p[k-1];
		}
		//System.out.println("Cumulated sum: "+p[p.length-1]);
		
		// Do a (scaled) sample from joint distribution because of unnormalized p's
		double u = Math.random() * p[p.length-1];
		int z_i=0;
		for(z_i=0; z_i<p.length; z_i++) { //!
			if(u<p[z_i]) {
				//System.out.println("X");
				break;
			}
		}
		*/
		
		// Sample z_i
		int z_i = FuncUtils.nextDiscrete(p);
		
		// Add newly sampled z_i to count variables
		//N[c][d][z_i] += 1;
		N[c][z_i][d] += 1;
		Nsum[c][d] += 1;
		if(z_i<K) { // Common topic
			MI[z_i][corpus.get(c).get(d).get(n)] += 1;
			MIsum[z_i] += 1;
		}
		else { // Private topic
			MS[c][z_i-K][corpus.get(c).get(d).get(n)] += 1;
			MSsum[c][z_i-K] += 1;
		}
		
		return z_i;
	}
	
	private void updateAlpha(int c) {
		int n_iter = 5;
		double shape = 1.001;
		double scale = 1.0;
		
		double alphaSum = 0;
		for(double a : alpha[c]) {
			alphaSum += a;
		}
		
		int[][] C_K = new int[K+K_C[c]][];
		for(int k=0; k<K+K_C[c]; k++) {
			C_K[k] = FuncUtils.countHist(N[c][k]);
		}
		int[] C_Sigma = FuncUtils.countHist(Nsum[c]);
		
		for(int i=0; i<n_iter; i++) {
			double denom = 0;
			double currDigamma = 0;
			for(int j=0; j<C_Sigma.length; j++) {
				if(C_Sigma[j]!=0) {
					currDigamma += 1.0/(alphaSum+j-1);
					denom += C_Sigma[j]*currDigamma;
				}
			}
			denom -= 1.0/scale;
			alphaSum = 0;
			for(int k=0; k<K+K_C[c]; k++) {
				double oldAlpha_k = alpha[c][k];
				double num = 0;
				currDigamma = 0;
				for(int j=0; j<C_K[k].length; j++) {
					if(C_K[k][j]!=0) {
						currDigamma += 1.0/(oldAlpha_k+j-1);
						num += C_K[k][j]*currDigamma;
					}
				}
				num += shape;
				alpha[c][k] = oldAlpha_k*(num/denom);
				alphaSum += alpha[c][k];
			}
		}
	}
	
	private void updateParams() 
	{
		
	}
	
	
	public static void main(String[] args)
	{
		
		// Load data_conf.json
		String[] datanames = {"All_Beauty","Luxury_Beauty"};
		String[] datadirs = {"20k","20k"};
		String datapath = "/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/Cross_collection/"
							+ datanames[0]+"&"+datanames[1]+"/"+datadirs[0]+"&"+datadirs[1]+"/";
		HashMap<String, String> dataConf = FileUtils.loadDataConf(datapath+"data_conf.json");
				
		// Define model parameters
		double alpha = 0.1;
		double beta = 0.01;
		int C=2;
		int K = 10;
		int[] K_C = {3, 4};
		
		C_LDA_simple clda = new C_LDA_simple(datanames, dataConf, C, K, K_C, alpha, beta, 1000);
		clda.inference();
		
		
		
		/**
		int[] counts = {0,0,0,0,0,0,1,1,1,2,2,3,3,4,5,5,5};
		IntSummaryStatistics stat = Arrays.stream(counts).summaryStatistics();
		int[] hist = new int[stat.getMax()+1];
		
		for(int c : counts) {
			hist[c] += 1;
		}
		
		for(int i=0; i<hist.length; i++) {
			System.out.println(hist[i]);
		}
		*/
	}
	
	
	
	
	
	
	
	
	
	
	
	
}
