package models;

import java.awt.desktop.AboutHandler;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import utility.FileUtils;
import utility.FuncUtils;

public class CAMEL {
	public double alpha; //Hyper-parameter alpha
	public double beta; //Hyper-parameter beta
	public double gamma; //Hyper-parameter gamma
	public int C; //Number of collections
	public int K; //Number of common topics
	public int[] K_C; // Numbers of private topics in each collection
	
	public String[] dataNames;
	public List<List<List<List<Integer>>>> corpus; // Word ID-based corpus
	public int[] D_C; // Number of documents in each collection
	
	public HashMap<String, Integer> token2id; // Vocabulary: word -> ID
	public HashMap<Integer, String> id2token; // Vocabulary: ID -> word
	public int V; // Vocabulary size (combined)
	
	int z[][][]; //z[c][d][s] = topic assignment of sentence w_cds
	int r[][][]; //r[c][d][s] = common-specific topic switcher of sentence w_cds
	int y[][][][]; //y[c][d][s][n] = topic-opinion switcher of word w_cdsn
	
	int[][][] NI; // N[c][d][k]=Number of sentences in document d assigned to common topic k
	int[][] NIsum; // Sum over N[c][d][k]'s, i.e., Number of sentences in document d assigned to common topics
	
	int[][][] NS; // NS[c][d][k] = Number of sentences in document d∈D_c assigned to private topic k∈K_c
	int[][] NSsum; // Sum over NS[c][d][k]'s, i.e., Number of sentences in document d∈D_c assigned to private topics
	
	int[][] MI; // MI[k][v] = Number of times common topic k∈K has been assigned to term v while v is treated as topic word
	int[] MIsum; // Sum over MI[k][v]'s, i.e., number of times any word is assigned as topic-word of common topic k∈K
	
	int[][][] MS; // MS[c][k][v] = Number of times private topic k∈K_c has been assigned to term v while v is treated as topic word
	int[][] MSsum; // Sum over MS[c][k][v]'s, i.e., number of times any word is assigned as topic-word of private topic k∈K_c
	
	int[][][] OI; // OI[c][k][v] = Number of times common topic k∈K has been assigned to term v in collection c while v is treated as opinion word
	int[][] OIsum; // Sum over OI[c][k][v]'s, i.e., number of times (in collection c) any word is assigned as opinion-word of common topic k∈K
	
	int[][][] OS; // OS[c][k][v] = Number of times private topic k∈K_c has been assigned to term v (in collection c) 
				  // while v is treated as opinion word
	int[][] OSsum; // Sum over OS[c][k][v]'s, i.e., number of times (in collection c) any word is assigned as opinion-word
				   // of private topic k∈K_c
	
	double[][][][] pi;
	
	private static int THIN_INTERVAL = 20; //sampling lag (?)
	private static int BURN_IN = 100; //burn-in period
	private static int ITERATIONS = 1000; //max iterations
	private static int dispcol = 0;
	private static int SAMPLE_LAG; //sample lag (if -1 only one sample taken)
	
	public CAMEL(String[] dataNames, HashMap<String, String> dataConf,
			int C, int K, int[] K_C, double alpha, double beta, double gamma)
	{
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
		this.C = C;
		this.K = K;
		this.K_C = K_C;
		this.dataNames = dataNames;
		
		System.out.println("Reading topic modeling corpus...");
		// Load vocabulary
		token2id = FileUtils.loadToken2idx(dataConf.get("token2idx_file"));
		id2token = FileUtils.loadIdx2token(dataConf.get("idx2token_file"));
		// Load corpus
		corpus = new ArrayList<List<List<List<Integer>>>>();
		for(int c=0; c<C; c++) {
			String key = dataNames[c]+"_train_file_vidx";
			List<List<List<Integer>>> arr = FileUtils.loadCorpusSentences(dataConf.get(key));
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
		
		OI = new int[C][K][V];
		OIsum = new int[C][K];
		
		OS = new int[C][][];
		for(int c=0; c<C; c++) { OS[c] = new int[K_C[c]][V]; }
		OSsum = new int[C][];
		for(int c=0; c<C; c++) { OSsum[c] = new int[K_C[c]]; }
		
		// Load topic-opinion switch probabilities
		pi = FuncUtils.loadMaxEntProbabilities();
		
		// Define hidden variables
		r = new int[C][][];
		z = new int[C][][];
		y = new int [C][][][];
		
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of common topics: " + K);
		for(int c=0; c<C; c++) {
			System.out.println(dataNames[c]+": Number of private topics: " + K_C[c]);
		}
		
		// Initialize hidden variables and count matrices
		initialize();
	}
	
	//Randomly initialize topic assignments and switchers (z's, r's and y's)
	public void initialize()
	{
		double[] switch_p = {0.5,0.5};
		double[] common_p = new double[K]; Arrays.fill(common_p, 1.0/K);
		double[][] private_p = new double[C][];
		for(int c=0; c<C; c++) {
			private_p[c] = new double[K_C[c]];
			Arrays.fill(common_p, 1.0/K_C[c]);
		}
		
		for(int c=0; c<C; c++) { // Loop collections
			r[c] = new int[D_C[c]][];
			z[c] = new int[D_C[c]][];
			y[c] = new int[D_C[c]][][];
			for(int d=0; d<D_C[c]; d++) { // Loop documents
				int N_d = corpus.get(c).get(d).size(); // Number of sentences in document d
				r[c][d] = new int[N_d];
				z[c][d] = new int[N_d];
				y[c][d] = new int[N_d][];
				for(int s=0; s<N_d; s++) { // Loop sentences
					int r_cds = FuncUtils.nextDiscrete(switch_p);
					int z_cds;
					if(r_cds==1) {// common topic
						z_cds = FuncUtils.nextDiscrete(common_p);
						// Increase counts
						NI[c][d][z_cds] += 1;
						NIsum[c][d] += 1;
					}
					else {// private topic
						z_cds = FuncUtils.nextDiscrete(private_p[c]);
						// Increase counts
						NS[c][d][z_cds] += 1;
						NSsum[c][d] += 1;
					}
					r[c][d][s] = r_cds;
					z[c][d][s] = z_cds;
					int N_s = corpus.get(c).get(d).get(s).size(); // Number of words in sentence s
					for(int n=0; n<N_s; n++) {
						int w_cdsn = corpus.get(c).get(d).get(s).get(n);
						int y_cdsn = FuncUtils.nextDiscrete(pi[c][d][s][n]);
						if(r_cds==1 && y_cdsn==0) { // common topic & topic-word
							MI[z_cds][w_cdsn] += 1;
							MIsum[z_cds] += 1;
						}
						else if(r_cds==1 && y_cdsn==1) { // common topic & opinion-word
							OI[c][z_cds][w_cdsn] += 1;
							OIsum[c][z_cds] += 1;
						}
						else if(r_cds==0 && y_cdsn==0) { // private topic & topic-word
							MS[c][z_cds][w_cdsn] += 1;
							MSsum[c][z_cds] += 1;
						}
						else { // private topic & opinion-word
							OS[c][z_cds][w_cdsn] += 1;
							OSsum[c][z_cds] += 1;
						}
						y[c][d][s][n] = y_cdsn;
					}
				}
			}
		}
		System.out.println("Initialization Done!");
	}
	
	public void inference(int n_iter)
	{
		ITERATIONS = n_iter;
		System.out.println("Running Gibbs sampling inference: ");
		System.out.println("Sampling "+ITERATIONS+" iterations with burn-in of "+ BURN_IN +" (B/S="+THIN_INTERVAL+").");
		
		int minDocs = Arrays.stream(D_C).min().getAsInt();
		for(int i=0; i<ITERATIONS; i++) {
			// Do sampling in following order: z[0][0][.],z[1][0][.], z[0][1][.],z[1][1][.], z[0][2][.],z[1][2][.], ...
			for(int d=0; d<minDocs; d++) {
				for(int c=0; c<C; c++) {
					for(int s=0; s<z[c][d].length; s++) {
						int[] sample = sampleFullConditionalBlock(c, d, s); // sample z_i,r_i as a block
						r[c][d][s] = sample[0];
						z[c][d][s] = sample[1];
						int N_s = corpus.get(c).get(d).get(s).size(); // Number of words in sentence s
						for(int n=0; n<N_s; n++) {
							int y_j = sampleFullConditional(c, d, s, n);
							y[c][d][s][n] = y_j;
						}
					}
				}
			}
			// Finish sampling for collections c where D_C[c]>minDocs
			for(int c=0; c<C; c++) {
				for(int d=minDocs; d<D_C[c]; d++) {
					for(int s=0; s<z[c][d].length; s++) {
						int[] sample = sampleFullConditionalBlock(c, d, s); // sample z_i,r_i as a block
						r[c][d][s] = sample[0];
						z[c][d][s] = sample[1];
						int N_s = corpus.get(c).get(d).get(s).size(); // Number of words in sentence s
						for(int n=0; n<N_s; n++) {
							int y_j = sampleFullConditional(c, d, s, n);
							y[c][d][s][n] = y_j;
						}
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
		System.out.println("Sampling completed!");
	}
	
	/**
	 * Sample z_i,r_i from full conditional p(z_i,r_i|Z_-i,R_-i), where i=(c,d,s)
	 * @param c: Collection indicator
	 * @param d: Document indicator
	 * @param s: Sentence indicator
	 * @return z_i, r_i
	 */
	private int[] sampleFullConditionalBlock(int c, int d, int s) 
	{
		// Update count variables, i.e., remove effect of current z_i and r_i
		// nt_no[0] = n_t(.,w_i), nt_no[1] = n_o(.,w_i)
		List<HashMap<Integer, Integer>> nt_no = FuncUtils.createCountTablesCAMEL(corpus.get(c).get(d).get(s), y[c][d][s]);
		int ntSigma = 0;
		int noSigma = 0;
		if(r[c][d][s]==0) { // z[c][d][s] is private topic
			NS[c][d][z[c][d][s]] -= 1;
			NSsum[c][d] -= 1;
			for(int v : nt_no.get(0).keySet()) {
				MS[c][z[c][d][s]][v] -= nt_no.get(0).get(v);
				ntSigma += nt_no.get(0).get(v);
			}
			MSsum[c][z[c][d][s]] -= ntSigma;
			for(int v : nt_no.get(1).keySet()) {
				OS[c][z[c][d][s]][v] -= nt_no.get(1).get(v);
				noSigma += nt_no.get(1).get(v);
			}
			OSsum[c][z[c][d][s]] -= noSigma;
		}
		else { // z[c][d][s] is common topic
			NI[c][d][z[c][d][s]] -= 1;
			NIsum[c][d] -= 1;
			for(int v : nt_no.get(0).keySet()) {
				MI[z[c][d][s]][v] -= nt_no.get(0).get(v);
				ntSigma += nt_no.get(0).get(v);
			}
			MIsum[z[c][d][s]] -= ntSigma;
			for(int v : nt_no.get(1).keySet()) {
				OI[c][z[c][d][s]][v] -= nt_no.get(1).get(v);
				noSigma += nt_no.get(1).get(v);
			}
			OIsum[c][z[c][d][s]] -= noSigma;
		}
		
		// Construct probability table
		double[] p = new double[K+K_C[c]];
		for(int k=0; k<K; k++) { // First half, r_i=1, i.e., common topic
			double num1 = 1.0;
			for(int v : nt_no.get(0).keySet()) {
				for(int j=1; j<=nt_no.get(0).get(v); j++) {
					num1 *= (MI[k][v]+beta+nt_no.get(0).get(v)-j);
				}
			}
			double denom1 = 1.0;
			for(int j=1; j<=ntSigma; j++) {
				denom1 *= (MIsum[k]+V*beta+ntSigma-j);
			}
			double num2 = 1.0;
			for(int v : nt_no.get(1).keySet()) {
				for(int j=1; j<=nt_no.get(1).get(v); j++) {
					num2 *= (OI[c][k][v]+beta+nt_no.get(1).get(v)-j);
				}
			}
			double denom2 = 1.0;
			for(int j=1; j<=noSigma; j++) {
				denom2 *= (OIsum[c][k]+V*beta+noSigma-j);
			}
			p[k] = (NIsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
				   (NI[c][d][k]+alpha)/(NIsum[c][d]+K*alpha) *
				   num1/denom1 * num2/denom2;	
		}
		
		for(int k=0; k<K_C[c]; k++) { // Second half, r_i=0, i.e., private topic
			double num1 = 1.0;
			for(int v : nt_no.get(0).keySet()) {
				for(int j=1; j<=nt_no.get(0).get(v); j++) {
					num1 *= (MS[c][k][v]+beta+nt_no.get(0).get(v)-j);
				}
			}
			double denom1 = 1.0;
			for(int j=1; j<ntSigma; j++) {
				denom1 *= (MSsum[c][k]+V*beta+ntSigma-j);
			}
			double num2 = 1.0;
			for(int v : nt_no.get(1).keySet()) {
				for(int j=1; j<nt_no.get(1).get(v); j++) {
					num2 *= (OS[c][k][v]+beta+nt_no.get(1).get(v)-j);
				}
			}
			double denom2 = 1.0;
			for(int j=1; j<=noSigma; j++) {
				denom2 *= (OSsum[c][k]+V*beta+noSigma-j);
			}
			p[K+k] = (NSsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
					 (NS[c][d][k]+alpha)/(NSsum[c][d]+K_C[c]*alpha) *
					 num1/denom1 * num2/denom2; 
		}
		
		// Sample new z_i, r_i
		int sample = FuncUtils.nextDiscrete(p);
		int z_i, r_i;
		if(sample<K) { // Common topic, i.e., z_i∈K, r_i=1
			z_i = sample;
			r_i = 1;
			// Update count matrices accordingly to newly sampled z_i, r_i
			NI[c][d][z_i] += 1;
			NIsum[c][d] += 1;
			for(int v : nt_no.get(0).keySet()) {
				MI[z_i][v] += nt_no.get(0).get(v);
			}
			MIsum[z_i] += ntSigma;
			for(int v : nt_no.get(1).keySet()) {
				OI[c][z_i][v] += nt_no.get(1).get(v);
			}
			OIsum[c][z_i] += noSigma;
		}
		else { // Private topic, i.e., z_i∈K_c, r_i=0
			z_i = sample-K;
			r_i = 0;
			// Update count matrices accordingly to newly sampled z_i, r_i
			NS[c][d][z_i] += 1;
			NSsum[c][d] += 1;
			for(int v : nt_no.get(0).keySet()) {
				MS[c][z_i][v] += nt_no.get(0).get(v);
			}
			MSsum[c][z_i] += ntSigma;
			for(int v : nt_no.get(1).keySet()) {
				OS[c][z_i][v] += nt_no.get(1).get(v);
			}
			OSsum[c][z_i] += noSigma;
		}
		
		int[] ri_zi = {r_i,z_i};
		return ri_zi;
	}
	
	private int[] sampleFullConditionalBlock2(int c, int d, int s)
	{
		// Update count variables, i.e., remove effect of current z_i and r_i
		List<Entry<Integer,Integer>>[] nt_no = FuncUtils.createCountTablesCAMEL2(corpus.get(c).get(d).get(s), y[c][d][s]);
		int ntSigma = 0;
		int noSigma = 0;
		if(r[c][d][s]==0) { // Current z[c][d][s] is private topic
			NS[c][d][z[c][d][s]] -= 1;
			NSsum[c][d] -= 1;
			for(Entry<Integer,Integer> nt : nt_no[0]) {
				MS[c][z[c][d][s]][nt.getKey()] -= nt.getValue();
				ntSigma += nt.getValue();
			}
			MSsum[c][z[c][d][s]] -= ntSigma;
			for(Entry<Integer,Integer> no : nt_no[1]) {
				OS[c][z[c][d][s]][no.getKey()] -= no.getValue();
				noSigma += no.getValue();
			}
			OSsum[c][z[c][d][s]] -= noSigma;
		}
		else { // Current z[c][d][s] is common topic
			NI[c][d][z[c][d][s]] -= 1;
			NIsum[c][d] -= 1;
			for(Entry<Integer,Integer> nt : nt_no[0]) {
				MI[z[c][d][s]][nt.getKey()] -= nt.getValue();
				ntSigma += nt.getValue();
			}
			MIsum[z[c][d][s]] -= ntSigma;
			for(Entry<Integer,Integer> no : nt_no[1]) {
				OI[c][z[c][d][s]][no.getKey()] -= no.getValue();
				noSigma += no.getValue();
			}
			OIsum[c][z[c][d][s]] -= noSigma;
		}
		// Construct probability table
		double[] p = new double[K+K_C[c]];
		for(int k=0; k<K; k++) { // First half, r_i=1, i.e., common topic
			double num1 = 1.0;
			for(Entry<Integer,Integer> nt : nt_no[0]) {
				for(int j=1; j<=nt.getValue(); j++) {
					num1 *= (MI[k][nt.getKey()]+beta+nt.getValue()-j);
				}
			}
			double denom1 = 1.0;
			for(int j=1; j<=ntSigma; j++) {
				denom1 *= (MIsum[k]+V*beta+ntSigma-j);
			}
			double num2 = 1.0;
			for(Entry<Integer,Integer> no : nt_no[1]) {
				for(int j=1; j<=no.getValue(); j++) {
					num2 *= (OI[c][k][no.getKey()]+beta+no.getValue()-j);
				}
			}
			double denom2 = 1.0;
			for(int j=1; j<=noSigma; j++) {
				denom2 *= (OIsum[c][k]+V*beta+noSigma-j);
			}
			p[k] = (NIsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
				   (NI[c][d][k]+alpha)/(NIsum[c][d]+K*alpha) *
				   num1/denom1 * num2/denom2;	
		}
		
		for(int k=0; k<K_C[c]; k++) { // Second half, r_i=0, i.e., private topic
			double num1 = 1.0;
			for(Entry<Integer,Integer> nt : nt_no[0]) {
				for(int j=1; j<=nt.getValue(); j++) {
					num1 *= (MS[c][k][nt.getKey()]+beta+nt.getValue()-j);
				}
			}
			double denom1 = 1.0;
			for(int j=1; j<ntSigma; j++) {
				denom1 *= (MSsum[c][k]+V*beta+ntSigma-j);
			}
			double num2 = 1.0;
			for(Entry<Integer,Integer> no : nt_no[1]) {
				for(int j=1; j<no.getValue(); j++) {
					num2 *= (OS[c][k][no.getKey()]+beta+no.getValue()-j);
				}
			}
			double denom2 = 1.0;
			for(int j=1; j<=noSigma; j++) {
				denom2 *= (OSsum[c][k]+V*beta+noSigma-j);
			}
			p[K+k] = (NSsum[c][d]+gamma)/(NIsum[c][d]+NSsum[c][d]+2*gamma) *
					 (NS[c][d][k]+alpha)/(NSsum[c][d]+K_C[c]*alpha) *
					 num1/denom1 * num2/denom2;
		}
		// Sample new z_i, r_i
		int sample = FuncUtils.nextDiscrete(p);
		int z_i, r_i;
		if(sample<K) { // Common topic, i.e., z_i∈K, r_i=1
			z_i = sample;
			r_i = 1;
			// Update count matrices accordingly to newly sampled z_i, r_i
			NI[c][d][z_i] += 1;
			NIsum[c][d] += 1;
			for(Entry<Integer,Integer> nt : nt_no[0]) {
				MI[z_i][nt.getKey()] += nt.getValue();
			}
			MIsum[z_i] += ntSigma;
			for(Entry<Integer,Integer> no : nt_no[1]) {
				OI[c][z_i][no.getKey()] += no.getValue();
			}
			OIsum[c][z_i] += noSigma;
		}
		else { // Private topic, i.e., z_i∈K_c, r_i=0
			z_i = sample-K;
			r_i = 0;
			// Update count matrices accordingly to newly sampled z_i, r_i
			NS[c][d][z_i] += 1;
			NSsum[c][d] += 1;
			for(Entry<Integer,Integer> nt : nt_no[0]) {
				MS[c][z_i][nt.getKey()] += nt.getValue();
			}
			MSsum[c][z_i] += ntSigma;
			for(Entry<Integer,Integer> no : nt_no[1]) {
				OS[c][z_i][no.getKey()] += no.getValue();
			}
			OSsum[c][z_i] += noSigma;
		}
		int[] ri_zi = {r_i,z_i};
		return ri_zi;
	}
	
	private int sampleFullConditional(int c, int d, int s, int n)
	{
		int r_i = r[c][d][s];
		int z_i = z[c][d][s];
		int w_j = corpus.get(c).get(d).get(s).get(n);
		// Update count variables, i.e., remove effect of current y_j
		if(r_i==1) { // common topic, i.e., r_i=1, z_i∈K
			if(y[c][d][s][n]==0) { // Current w_j is topic word
				MI[z_i][w_j] -= 1;
				MIsum[z_i] -= 1;
			}
			else { // Current w_j is opinion word
				OI[c][z_i][w_j] -= 1;
				OIsum[c][z_i] -= 1;
			}
			// Construct probability table
			double[] p = new double[2];
			p[0] = (1-pi[c][d][s][n]) * (MI[z_i][w_j]+beta)/(MIsum[z_i]+V*beta);
			p[1] = pi[c][d][s][n] * (OI[c][z_i][w_j]+beta)/(OIsum[c][z_i]+V*beta);
			// Sample new y_j
			int y_j = FuncUtils.nextDiscrete(p);
			//int y_j = FuncUtils.nextDiscrete(p[0]);
			
			// Add newly sampled y_j count variables
			if(y_j==0) { // topic word
				MI[z_i][w_j] += 1;
				MIsum[z_i] += 1;
			}
			else { // opinion word
				OI[c][z_i][w_j] += 1;
				OIsum[c][z_i] += 1;
			}
			return y_j;
			
		}
		else { // private topic, i.e., r_i=0, z_i∈K_c
			if(y[c][d][s][n]==0) { // Current w_j is topic word
				MS[c][z_i][w_j] -= 1;
				MSsum[c][z_i] -= 1;
			}
			else { // Current w_j is opinion word
				OS[c][z_i][w_j] -= 1;
				OSsum[c][z_i] -= 1;
			}
			// Construct probability table
			double[] p = new double[2];
			p[0] = (1-pi[c][d][s][n]) * (MS[c][z_i][w_j]+beta)/(MSsum[c][z_i]+V*beta);
			p[1] = pi[c][d][s][n] * (OS[c][z_i][w_j]+beta)/(OSsum[c][z_i]+V*beta);
			// Sample new y_j
			int y_j = FuncUtils.nextDiscrete(p);
			//int y_j = FuncUtils.nextDiscrete(p[0]);
			
			// Add newly sampled y_j count variables
			if(y_j==0) { // topic word
				MS[c][z_i][w_j] += 1;
				MSsum[c][z_i] += 1;
			}
			else { // opinion word
				OS[c][z_i][w_j] += 1;
				OSsum[c][z_i] += 1;
			}
			return y_j;
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
		double gamma = 0.1;
		int C=2;
		int K = 10;
		int[] K_C = {4, 4};
	}
	
}
