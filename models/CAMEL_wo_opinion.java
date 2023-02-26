package models;

import java.awt.RenderingHints.Key;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Arrays;

import javax.sql.rowset.JoinRowSet;

import utility.FileUtils;
import utility.FuncUtils;

import java.util.ArrayList;

public class CAMEL_wo_opinion {
	public double alpha; //Hyper-parameter alpha
	public double beta; //Hyper-parameter beta
	public double gamma; //Hyper-parameter gamma
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
	int r[][][]; //r[c][d][n] = common-specific topic switcher of word w_cdn
	
	int[][][] NI; // NI[c][d][k] = Number of words in document d of collection c assigned to common topic k.
	int[][] NIsum; // NIsum.get[c][d] = Sum over NI[c][d][k]'s, i.e., number of words in document d of collection c assigned to common topics.
	int[][][] NS; // NS[c][d][k] = Number of words in document d from collection c assigned to private topic k.
	int[][] NSsum; // NSsum[c][d] = Sum over NS[c][d][k]'s, i.e., number of words in document d from collection c assigned to private topics. *
	int[][] MI; // MI[k][v] = Number of times common topic k has been assigned to term v
	int[] MIsum; // MIsum[k] = Sum over MI[k][v]'s, i.e., total number of words assigned to common topic k.
	int[][][] MS; // MS[c][k][v] = Number of times private topic k of collection c has been assigned to term v.
	int[][] MSsum; // MSsum[c][k] = Sum over MS[c][k][v]'s, i.e., total number of words assigned to private topic k of collection c.
	
	private static int THIN_INTERVAL = 20; //sampling lag (?)
	private static int BURN_IN = 100; //burn-in period
	private static int ITERATIONS = 1000; //max iterations
	private static int dispcol = 0;
	private static int SAMPLE_LAG; //sample lag (if -1 only one sample taken)
	
	public double initTime = 0; // Initialization time
	public double iterTime = 0; // Gibbs sampling time
	
	public CAMEL_wo_opinion(String[] dataNames, HashMap<String, String> dataConf,
			int C, int K, int[] K_C, double alpha, double beta, double gamma,
			int numIterations)//throws Exception
	{
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
		this.C = C;
		this.K = K;
		this.K_C = K_C;
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
		NI = new int[C][][];
		for(int c=0; c<C; c++) { NI[c] = new int[D_C[c]][K]; }
		NIsum = new int[C][];
		for(int c=0; c<C; c++) { NIsum[c] = new int[D_C[c]]; }
		NS = new int[C][][];
		for(int c=0; c<C; c++) { NS[c] = new int[D_C[c]][K_C[c]]; }
		NSsum = new int[C][];
		for(int c=0; c<C; c++) { NSsum[c] = new int[D_C[c]]; }
		MI = new int[K][V];
		MIsum = new int[K];
		MS = new int[C][][];
		for(int c=0; c<C; c++) { MS[c] = new int[K_C[c]][V]; }
		MSsum = new int[C][];
		for(int c=0; c<C; c++) { MSsum[c] = new int[K_C[c]]; }
		
		// Initialize hidden variables
		r = new int[C][][];
		z = new int[C][][];
		
		//System.out.println("Corpus size: "++" docs, "++" words");
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of common topics: " + K);
		for(int c=0; c<C; c++) {
			System.out.println(dataNames[c]+": Number of private topics: " + K_C[c]);
		}
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("gamma: " + gamma);
		System.out.println("Number of sampling iterations: " + ITERATIONS);
		
		initialize();
	}
	
	//Randomly initialize topic assignments and switchers (z's and r's)
	public void initialize()//throws IOException
	{
		System.out.println("Randomly initializing hidden variables ...");
		
		double[] switch_p = {0.5,0.5};
		double[] common_p = new double[K]; 
		for(int i=0; i<K; i++) {
			common_p[i] = 1.0/K;
		}
		double[][] private_p = new double[C][];
		for(int c=0; c<C; c++) {
			private_p[c] = new double[K_C[c]];
			for(int k=0; k<K_C[c]; k++) {
				private_p[c][k] = 1.0/K_C[c];
			}
		}
		
		long startTime = System.currentTimeMillis();
		for(int c=0; c<C; c++) {
			r[c] = new int[D_C[c]][];
			z[c] = new int[D_C[c]][];
			for(int d=0; d<D_C[c]; d++) {
				int docSize = corpus.get(c).get(d).size();
				r[c][d] = new int[docSize];
				z[c][d] = new int[docSize];
				for(int n=0; n<docSize; n++) {
					int r_cdn = FuncUtils.nextDiscrete(switch_p);
					int z_cdn;
					if(r_cdn==1) { // common topic
						z_cdn = FuncUtils.nextDiscrete(common_p);
						// Increase counts
						NI[c][d][z_cdn] += 1;
						NIsum[c][d] += 1;
						MI[z_cdn][corpus.get(c).get(d).get(n)] += 1;
						MIsum[z_cdn] += 1;
					}
					else { // private topic
						z_cdn = FuncUtils.nextDiscrete(private_p[c]);
						NS[c][d][z_cdn] += 1;
						NSsum[c][d] += 1;
						MS[c][z_cdn][corpus.get(c).get(d).get(n)] += 1;
						MSsum[c][z_cdn] += 1;
					}
					r[c][d][n] = r_cdn;
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
						int[] sample = sampleFullConditionalBlock(c, d, n);
						r[c][d][n] = sample[0];
						z[c][d][n] = sample[1];
					}
				}
			}
			// Finish sampling for collections c where D_C[c]>minDocs
			for(int c=0; c<C; c++) {
				for(int d=minDocs; d<D_C[c]; d++) {
					for(int n=0; n<z[c][d].length; n++) {
						int[] sample = sampleFullConditionalBlock(c, d, n);
						r[c][d][n] = sample[0];
						z[c][d][n] = sample[1];
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
		//iterTime = System.currentTimeMillis()-startTime;
		System.out.println("Sampling completed!");
	}
	
	/**
	 * Sample z_i,r_i from full conditional p(z_i,r_i|Z_-i,R_-i), where i=(c,d,n)
	 * @param c: Collection indicator
	 * @param d: Document indicator
	 * @param n: Word indicator
	 * @return z_i,r_i
	 */
	private int[] sampleFullConditionalBlock(int c, int d, int n)
	{
		// Update count variables
		if(r[c][d][n]==0) { // z[c][d][n] is private topic
			NS[c][d][z[c][d][n]] -= 1;
			NSsum[c][d] -= 1;
			MS[c][z[c][d][n]][corpus.get(c).get(d).get(n)] -=1;
			MSsum[c][z[c][d][n]] -= 1;
		}
		else { // z[c][d][n] is common topic
			NI[c][d][z[c][d][n]] -= 1;
			NIsum[c][d] -= 1;
			MI[z[c][d][n]][corpus.get(c).get(d).get(n)] -= 1;
			MIsum[z[c][d][n]] -= 1;
		}
		
		// Construct 2x(K+K_c) unnormalized probability table in 1x(2*(K+K_c)) vector
		double[] p = new double[K_C[c]+K];
		for(int k=0; k<K_C[c]; k++) { // First half, r=0, i.e., private topic
			//p[k] = (NSsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
				//	(NS[c][d][k]+alpha)/(NSsum[c][d]+K_C[c]*alpha) *
				//	(MS[c][k][corpus.get(c).get(d).get(n)]+beta)/(MSsum[c][k]+V*beta);
			p[k] = (NSsum[c][d]+gamma)*
					(NS[c][d][k]+alpha)/(NSsum[c][d]+K_C[c]*alpha) *
					(MS[c][k][corpus.get(c).get(d).get(n)]+beta)/(MSsum[c][k]+V*beta);
		}
		for(int k=0; k<K; k++) { // Second half, r=1, i.e., common topic
			//p[k+K_C[c]] = (NIsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
				//			(NI[c][d][k]+alpha)/(NIsum[c][d]+K*alpha) *
					//		(MI[k][corpus.get(c).get(d).get(n)]+beta)/(MIsum[k]+V*beta);
			p[k+K_C[c]] = (NIsum[c][d]+gamma)*
							(NI[c][d][k]+alpha)/(NIsum[c][d]+K*alpha) *
							(MI[k][corpus.get(c).get(d).get(n)]+beta)/(MIsum[k]+V*beta);
		}
		
		// Cumulate probabilities
		for(int k=1; k<p.length; k++) {
			p[k] += p[k-1];
		}
		//System.out.println("Cumulated sum: "+p[p.length-1]);
		
		// Do a (scaled) sample from joint distribution because of unnormalized p's
		double u = Math.random() * p[K+K_C[c]-1];
		int r_i,z_i=0;
		int[] rsize = {K_C[c],K};
		outer: for(r_i=0; r_i<2; r_i++) {
			for(z_i=0; z_i<rsize[r_i]; z_i++) {
				if(u<p[r_i*K_C[c]+z_i])
					break outer;
			}
		}
		
		// Add newly sampled r_i and z_i to count variables
		if(r_i==0) { // z_i is private topic
			NS[c][d][z_i] += 1;
			NSsum[c][d] += 1;
			MS[c][z_i][corpus.get(c).get(d).get(n)] +=1;
			MSsum[c][z_i] += 1;
		}
		else { // z_i is common topic
			NI[c][d][z_i] += 1;
			NIsum[c][d] += 1;
			MI[z_i][corpus.get(c).get(d).get(n)] += 1;
			MIsum[z_i] += 1;
		}
		
		int[] sample = {r_i,z_i};
		return sample;
	}
	
	private void updateParams() 
	{
		
	}
	
	public void writeTopTopicalWords(String savePath, String modelName, int topWords)throws IOException
	{
		File dir = new File(savePath);
		if (!dir.exists())
			dir.mkdir();
		
		// For common topics
		FileUtils.writeTopTopicalWords(MI, savePath+modelName+"_common.topWords",topWords,id2token);
		// For each collection private topics
		for(int c=0; c<C; c++) {
			FileUtils.writeTopTopicalWords(MS[c], savePath+modelName+"_private_"+dataNames[c]+".topWords",topWords,id2token);
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	public static void main(String[] args) {
		
		// Load data_conf.json
		String[] datanames = {"All_Beauty","Luxury_Beauty"};
		String[] datadirs = {"20k","20k"};
		String datapath = "/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/Cross_collection/"
							+ datanames[0]+"&"+datanames[1]+"/"+datadirs[0]+"&"+datadirs[1]+"/";
		HashMap<String, String> dataConf = FileUtils.loadDataConf(datapath+"data_conf.json");
		
		// Define model parameters
		double alpha = 0.1;
		double beta = 0.01;
		double gamma = 0.1;
		int C=2;
		int K = 10;
		int[] K_C = {4, 4};
		
		CAMEL_wo_opinion camel = new CAMEL_wo_opinion(datanames, dataConf, C, K, K_C, alpha, beta, gamma, 1000);
		camel.inference();
		
		/*
		for(int c=0; c<C; c++) {
			String key = datanames[c]+"_size";
			System.out.println(Integer.parseInt(dataConf.get(key)));
		}
		*/
		/*
		double[] p={0.1,0.9};
		for(int i=0; i<10; i++) {
			boolean r = (FuncUtils.nextDiscrete(p) != 0);
			//int r = FuncUtils.nextDiscrete(p);
			System.out.print(r);
		}
		*/
		
		/*
		double[] p1 = new double[K+K_C[c]];
		for(int k=0; k<K+K_C[c]; k++) { // r=0, i.e., private topic
			p1[k] = (NSsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
						(NS[c][d][k]+alpha)/(NSsum[c][d]+K_C[c]*alpha) *
						(MS[c][k][corpus.get(c).get(d).get(n)]+beta)/(MSsum[c][k]+V*beta);
		}
		double[] p2 = new double[K+K_C[c]];
		for(int k=0; k<K+K_C[c]; k++) { // r=1, i.e., common topic
			p2[k] = (NIsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
						(NI[c][d][k]+alpha)/(NIsum[c][d]+K*alpha) *
						(MI[k][corpus.get(c).get(d).get(n)]+beta)/(MIsum[k]+V*beta);
		}
		*/
	}

}
