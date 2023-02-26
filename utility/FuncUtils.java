package utility;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.IntSummaryStatistics;

public class FuncUtils 
{
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValueDescending(Map<K, V> map)
	{
		List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<K, V>>()
        {
            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2)
            {
                int compare = (o1.getValue()).compareTo(o2.getValue());
                return -compare;
            }
        });
		
		Map<K, V> result = new LinkedHashMap<K, V>();
		for(Map.Entry<K, V> entry : list) {
			result.put(entry.getKey(), entry.getValue());
		}
		return result;
	}
	
	/**
	 * Sample a value from a double array
	 * @param probs
	 * @return
	 */
	public static int nextDiscrete(double[] probs)
	{
		double sum = 0.0;
		for(int i=0; i<probs.length; i++) {
			sum += probs[i];
		}
		double r = MTRandom.nextDouble() * sum;
		
		sum = 0.0;
		for(int i=0; i<probs.length; i++) {
			sum += probs[i];
			if(sum>r)
				return i;
		}
		return probs.length-1;
	}
	
	public static int nextDiscrete(double prob) {
		double[] p = {prob, 1-prob};
		return nextDiscrete(p);
	}
	
	public static int[] countHist(int[] counts) 
	{
		IntSummaryStatistics stat = Arrays.stream(counts).summaryStatistics();
		int[] hist = new int[stat.getMax()+1];
		//System.out.println(stat.getMin());
		//System.out.println(stat.getMax());
		//System.out.println(hist.length);
		for(int c : counts) {
			hist[c] += 1;
		}
		
		return hist;
	}
	
	public static HashMap<Integer, Integer> createCountTable(List<Integer> list)
	{
		HashMap<Integer, Integer> hist = new HashMap<Integer, Integer>();
		for(int i=0; i<list.size(); i++) {
			if(!hist.containsKey(list.get(i)))
				hist.put(list.get(i), 1);
			else
				hist.put(list.get(i), hist.get(list.get(i))+1);
		}
		return hist;
	}
	
	/**
	 * Create count tables n_t(.,w_i) and n_o(.,w_i)
	 * @param w_i = Sentence
	 * @param y_i = Topic-opinion indicators for words in sentence w_i
	 * @return n_t(.,w_i), n_o(.,w_i)
	 */
	public static List<HashMap<Integer, Integer>> createCountTablesCAMEL(List<Integer> w_i, int[] y_i)
	{
		HashMap<Integer, Integer> nt = new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> no = new HashMap<Integer, Integer>();
		for(int j=0; j<w_i.size(); j++) {
			if(y_i[j]==0) { // w_i[j] is opinion word
				if(!no.containsKey(w_i.get(j)))
					no.put(w_i.get(j), 1);
				else
					no.put(w_i.get(j), no.get(w_i.get(j)+1));
			}
			else { // w_i[j] is topic word
				if(!nt.containsKey(w_i.get(j)))
					nt.put(w_i.get(j), 1);
				else 
					nt.put(w_i.get(j), nt.get(w_i.get(j)+1));
			}
		}
				
		List<HashMap<Integer, Integer>> tables = new ArrayList<HashMap<Integer, Integer>>();
		tables.add(nt);
		tables.add(no);
		
		return tables;
	}
	
	public static List<Entry<Integer,Integer>>[] createCountTablesCAMEL2(List<Integer> w_i, int[] y_i)
	{
		HashMap<Integer, Integer> nt = new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> no = new HashMap<Integer, Integer>();
		for(int j=0; j<w_i.size(); j++) {
			if(y_i[j]==0) { // w_i[j] is opinion word
				if(!no.containsKey(w_i.get(j)))
					no.put(w_i.get(j), 1);
				else
					no.put(w_i.get(j), no.get(w_i.get(j)+1));
			}
			else { // w_i[j] is topic word
				if(!nt.containsKey(w_i.get(j)))
					nt.put(w_i.get(j), 1);
				else 
					nt.put(w_i.get(j), nt.get(w_i.get(j)+1));
			}
		}
		
		List<Entry<Integer,Integer>>[] nt_no = new ArrayList[2];
		nt_no[0] = new ArrayList<>(nt.entrySet());
		nt_no[1] = new ArrayList<>(no.entrySet());
		//List<Entry<Integer,Integer>> nt_ = new ArrayList<>(nt.entrySet());
		//List<Entry<Integer,Integer>> no_ = new ArrayList<>(no.entrySet());
		
		//Integer[][][] nt_no = new Integer[2][2][];
		//nt_no[0][0] = nt.keySet().toArray(new Integer[0]);
		//nt_no[0][1] = nt.values().toArray(new Integer[0]);
		//nt_no[1][0] = no.keySet().toArray(new Integer[0]);
		//nt_no[1][1] = no.keySet().toArray(new Integer[0]);
		
		return nt_no;
	}
	
	public static double[][][][] loadMaxEntProbabilities(){
		
	}
	
}
