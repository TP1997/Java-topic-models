package models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import utility.FileUtils;
import utility.FuncUtils;

public class C_LDA {
	public double[][] alpha;
	public double beta;
	public double[] delta;
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
	int y[][][]; //y[c][d][n] = topic-word asymmetry indicator of word w_cdn
	
	int[][][] N; // (N[c][d][k]) N[c][k][d] = Number of words in document d of collection c assigned to topic k∈{1...K...K_c}
	int[][] Nsum; // Sum over N[c][d][k*]'s, i.e., total number of words in document d∈D_c
	
	int[][] M1; // M1[k][v] = Number of times common topic k∈{1,...,K} has been assigned to term v∈V while y=1
	int[] M1sum; // Sum over M1[k][v*]'s, i.e., number of times y=1 has been observed with common topic k∈{1,...,K}
	
	int[][][] M0; // M0[c][k][v] = Number of times common topic k∈{1,...,K} has been assigned to term v∈V while y=0 in collection c∈C
	int[][] M0sum; // Sum over M0[c][k][v*]'s, i.e, number of times y=0 has been observed with common topic k∈{1,...,K} in collection c∈C
	int[] M0sumsum; // Sum over M0[c*][k]'s, i.e., number if times y=0 has been observed with common topic k∈{1,...,K}
	
	int[][][] M; // M[c][k][v] = Number of times private topic k∈K_c/{K} has been assigned to term v in collection c∈C
	int[][] Msum; // Sum over M[c][k][v*]'s, i.e., number of times private topic k∈K_c/{K} has been observed in collection c∈C
	
	private static int THIN_INTERVAL = 10; //sampling lag (?)
	private static int BURN_IN = 100; //burn-in period
	private static int ITERATIONS = 1000; //max iterations
	private static int dispcol = 0;
	private static int SAMPLE_LAG = 10; //sample lag (if -1 only one sample taken)
	
	public double initTime = 0; // Initialization time
	public double iterTime = 0; // Gibbs sampling time
	
	public C_LDA(String[] dataNames, HashMap<String, String> dataConf,
			int C, int K, int[] K_C, double alpha, double beta, double[] delta)
	{
		this.beta = beta;
		this.delta = delta;
		this.C = C;
		this.K = K;
		this.K_C = K_C;
		initAlpha(alpha);
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
		
		M1 = new int[K][V];
		M1sum = new int[K];
		
		M0 = new int[C][K][V];
		M0sum  = new int[K][V];
		M0sumsum  = new int[K];
		
		M = new int[C][][];
		for(int c=0; c<C; c++) { M[c] = new int[K_C[c]][V]; }
		Msum = new int[C][];
		for(int c=0; c<C; c++) { Msum[c] = new int[K_C[c]]; }
		
		// Initialize hidden variables
		z = new int[C][][];
		y = new int[C][][];
		
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of common topics: " + K);
		for(int c=0; c<C; c++) {
			System.out.println(dataNames[c]+": Number of private topics: " + K_C[c]);
		}
		
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
	
	//Randomly initialize topic assignments and switchers (z's and y's)
	public void initialize()
	{
		System.out.println("Randomly initializing hidden variables ...");
		
		double[] switch_p = {0.5,0.5};
		double[][] p = new double[C][];
		for(int c=0; c<C; c++) {
			p[c] = new double[K+K_C[c]];
			for(int k=0; k<K+K_C[c]; k++) {
				p[c][k] = 1.0/(K+K_C[c]);
			}
		}
		
		for(int c=0; c<C; c++) {
			z[c] = new int[D_C[c]][];
			y[c] = new int[D_C[c]][];
			for(int d=0; d<D_C[c]; d++) {
				int docSize = corpus.get(c).get(d).size();
				z[c][d] = new int[docSize];
				y[c][d] = new int[docSize];
				for(int n=0; n<docSize; n++) {
					int z_cdn = FuncUtils.nextDiscrete(p[c]);
					int y_cdn = -1;
					// Increase counts
					//N[c][d][z_cdn] += 1;
					N[c][z_cdn][d] += 1;
					Nsum[c][d] += 1;
					if(z_cdn<K) { // common topic
						y_cdn = FuncUtils.nextDiscrete(switch_p);
						if(y_cdn==1) { // common w-dist
							M1[z_cdn][corpus.get(c).get(d).get(n)] += 1;
							M1sum[z_cdn] += 1;
						}
						else { // private w-dist
							M0[c][z_cdn][corpus.get(c).get(d).get(n)] += 1;
							M0sum[c][z_cdn] += 1;
							M0sumsum[z_cdn] += 1;
						}
					}
					else { // private topic
						M[c][z_cdn-K][corpus.get(c).get(d).get(n)] += 1;
						Msum[c][z_cdn-K] += 1;
					}
					
					z[c][d][n] = z_cdn;
					y[c][d][n] = y_cdn;
				}
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
		
		int minDocs = Arrays.stream(D_C).min().getAsInt();
		for(int i=0; i<ITERATIONS; i++) {
			// Do sampling in following order: z[0][0][.],z[1][0][.], z[0][1][.],z[1][1][.], z[0][2][.],z[1][2][.], ...
			for(int d=0; d<minDocs; d++) {
				for(int c=0; c<C; c++) {
					for(int n=0; n<z[c][d].length; n++) {
						int[] zy = sampleFullConditionalBlock(c, d, n);
						z[c][d][n] = zy[0];
						y[c][d][n] = zy[1];
					}
				}
			}
			// Finish sampling for collections c where D_C[c]>minDocs
			for(int c=0; c<C; c++) {
				for(int d=minDocs; d<D_C[c]; d++) {
					for(int n=0; n<z[c][d].length; n++) {
						int[] zy = sampleFullConditionalBlock(c, d, n);
						z[c][d][n] = zy[0];
						y[c][d][n] = zy[1];
					}
				}
			}
			
			if((i<BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.println("Burn-in");
				dispcol++;
			}
			if((i>BURN_IN) && (i%THIN_INTERVAL==0)) {
				System.out.println("Sampling:"+i+"/"+ITERATIONS);
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
	
	/**
	 * Sample z_i,r_i from full conditional p(z_i,y_i|Z_-i,Y_-i), where i=(c,d,n)
	 * @param c: Collection indicator
	 * @param d: Document indicator
	 * @param n: Word indicator
	 * @return z_i,y_i
	 */
	private int[] sampleFullConditionalBlock(int c, int d, int n)
	{
		int w = corpus.get(c).get(d).get(n);
		// Update count variables
		//N[c][d][z[c][d][n]] -= 1;
		N[c][z[c][d][n]][d] -= 1;
		Nsum[c][d] -= 1;
		if(z[c][d][n]<K) { // common topic
			if(y[c][d][n]==1) { // common w-dist
				M1[z[c][d][n]][w] -= 1;
				M1sum[z[c][d][n]] -= 1;
			}
			else { // private w-dist
				M0[c][z[c][d][n]][w] -= 1;
				M0sum[c][z[c][d][n]] -= 1;
				M0sumsum[z[c][d][n]] -= 1;
			}
		}
		else { // private topic
			M[c][z[c][d][n]-K][w] -= 1;
			Msum[c][z[c][d][n]-K] -= 1;
		}
		
		// Construct probability table for GS
		double[] p = new double[K+K+K_C[c]];
		// p(z_i,y_i=1|Z_-i,Y_-i) where z_i≤K
		for(int k=0; k<K; k++) {
			p[k] = (N[c][k][d]+alpha[c][k]) * (M1sum[k]+delta[0])/(M1sum[k]+delta[0]+M0sumsum[k]+delta[1]) * (M1[k][w]+beta)/(M1sum[k]+V*beta);
		}
		// p(z_i,y_i=0|Z_-i,Y_-i) where z_i≤K
		for(int k=0; k<K; k++) {
			p[K+k] = (N[c][k][d]+alpha[c][k]) * (M0sumsum[k]+delta[1])/(M1sum[k]+delta[0]+M0sumsum[k]+delta[1]) * (M0[c][k][w]+beta)/(M0sum[c][k]+V*beta);
		}
		// p(z_i,y_i|Z_-i,Y_-i)=p(z_i|Z_-i,Y_-i) where K < z_i ≤ K_c
		for(int k=0; k<K_C[c]; k++) {
			p[2*K+k] = (N[c][k][d]+alpha[c][k]) * (M[c][k][w]+beta)/(Msum[c][k]+V*beta);
		}
		
		/*
		// Cumulate probabilities
		for(int k=1; k<p.length; k++) {
			p[k] += p[k-1];
		}
		
		// Do a (scaled) sample from joint distribution because of unnormalized p's
		double u = Math.random() * p[p.length-1];
		int y_i,z_i=0;
		int[] rsize = {K,K,K_C[c]};
		outer: for(y_i=0; y_i<rsize.length; y_i++) {
			for(z_i=0; z_i<rsize[y_i]; z_i++) {
				if(u<p[y_i*K+z_i])
			}
		}
		*/
		
		int sample = FuncUtils.nextDiscrete(p);
		int z_i;//, y_i;
		// Add newly sampled z_i and y_i to count variables
		//N[c][][d] -= 1;
		//Nsum[c][d] -= 1;
		if(sample<K) { // common topic, common w-dist, i.e., z_i≤K, y_i=1
			z_i=sample; //y_i=1;
			N[c][z_i][d] += 1;
			Nsum[c][d] += 1;
			M1[z_i][w] += 1;
			M1sum[z_i] += 1;
			int[] zy = {z_i,1};
			return zy;
		}
		else if(sample<2*K) { // common topic, private w-dist, i.e., z_i≤K, y_i=0
			z_i=sample-K; //y_i=0;
			N[c][z_i][d] += 1;
			Nsum[c][d] += 1;
			M0[c][z_i][w] += 1;
			M0sum[c][z_i] += 1;
			M0sumsum[z_i] += 1;
			int[] zy = {z_i,0};
			return zy;
		}
		else { // private topic, i.e., K<z_i≤K_c
			z_i=sample-K;
			N[c][z_i][d] += 1;
			Nsum[c][d] += 1;
			M[c][z_i-K][w] -= 1;
			Msum[c][z_i-K] -= 1;
			int[] zy = {z_i,-1};
			return zy;
		}
		
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
		double[] delta = {1,1};
		int C=2;
		int K = 50;
		int[] K_C = {3, 4};
		
		C_LDA clda = new C_LDA(datanames, dataConf, C, K, K_C, alpha, beta, delta);
		clda.inference(1000);
	}
	
	
	
	
	
		
	
	
	
	
	
	
	
	
	
	
	
	
	
}
