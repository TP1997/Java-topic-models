package utility;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;
import java.util.Set;
//import javafx.util.Pair;
import java.util.TreeMap;

import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

import java.util.HashMap;

public class FileUtils {
	
	public static HashMap<String, String> loadDataConf(String filepath)
	{
		String jsonString = new String();
		try {
			jsonString = new String(Files.readAllBytes(Paths.get(filepath)));			
		} catch (Exception e) {
			e.printStackTrace();
		}
		HashMap<String, String> dataConf = new Gson().fromJson(jsonString, new TypeToken<HashMap<String, String>>() {}.getType());
		return dataConf;
	}
	
	public static HashMap<String, Integer> loadToken2idx(String filepath){
		String jsonString = new String();
		try {
			jsonString = new String(Files.readAllBytes(Paths.get(filepath)));			
		} catch (Exception e) {
			e.printStackTrace();
		}
		HashMap<String, Integer> word2IdVocabulary = new Gson().fromJson(jsonString, new TypeToken<HashMap<String, Integer>>() {}.getType());
		return word2IdVocabulary;
	}
	
	public static HashMap<Integer, String> loadIdx2token(String filepath){
		String jsonString = new String();
		try {
			jsonString = new String(Files.readAllBytes(Paths.get(filepath)));			
		} catch (Exception e) {
			e.printStackTrace();
		}
		HashMap<String, String> word2IdVocabulary = new Gson().fromJson(jsonString, new TypeToken<HashMap<String, String>>() {}.getType());
		
		HashMap<Integer, String> word2IdVocabulary_int = new HashMap<Integer, String>();
		for(Map.Entry<String,String> entry : word2IdVocabulary.entrySet()) {
			word2IdVocabulary_int.put(Integer.parseInt(entry.getKey()), entry.getValue());
		}
		
		return word2IdVocabulary_int;
	}
	
	public static ArrayList<List<Integer>> loadCorpus(String filepath){
		ArrayList<List<Integer>> corpus = new ArrayList<List<Integer>>();
		try {
			Scanner s1 = new Scanner(new File(filepath));
			while(s1.hasNextLine()) {
				String doc_vidx_str = s1.nextLine();
				if(doc_vidx_str.length()>0) {
					List<Integer> doc_vidx = new ArrayList<Integer>();
					for(String n : doc_vidx_str.split("\\s+")) {
						doc_vidx.add(Integer.parseInt(n));
					}
					corpus.add(doc_vidx);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return corpus;
	}
	public static ArrayList<List<List<Integer>>> loadCorpusSentences(String filepath){
		ArrayList<List<List<Integer>>> corpus = new ArrayList<List<List<Integer>>>();
		try {
			List<List<Integer>> doc_vidx = new ArrayList<List<Integer>>();
			Scanner s1 = new Scanner(new File(filepath));
			while(s1.hasNextLine()) {
				String sentence_vidx_str = s1.nextLine();
				if(sentence_vidx_str.length()>0) { // Sentence line
					List<Integer> sentence_vidx = new ArrayList<Integer>();
					for(String n : sentence_vidx_str.split("\\s+")) {
						sentence_vidx.add(Integer.parseInt(n));
					}
					doc_vidx.add(sentence_vidx);
				}
				else { // Empty line, document separator
					corpus.add(doc_vidx);
					doc_vidx = new ArrayList<List<Integer>>();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return corpus;
	}
	
	public static ArrayList<List<List<int[]>>> loadCountTable(String filepath){
		ArrayList<List<List<int[]>>> countTable = new ArrayList<List<List<int[]>>>();
		try {
			List<List<int[]>> doc_ct = new ArrayList<List<int[]>>();
			Scanner s1 = new Scanner(new File(filepath));
			while(s1.hasNextLine()) {
				String sentence_vidx_cnt_str = s1.nextLine();
				if(sentence_vidx_cnt_str.length()>0) { // Sentence line
					List<int[]> sentence_vidx_cnt = new ArrayList<int[]>();
					for(String n : sentence_vidx_cnt_str.split("\\s+")) {
						String[] entry = n.split(":");
						int vidx = Integer.parseInt(entry[0]);
						int cnt = Integer.parseInt(entry[1]);
						int[] vidx_cnt = {Integer.parseInt(entry[0]), Integer.parseInt(entry[1])};
						sentence_vidx_cnt.add(vidx_cnt);
					}
					doc_ct.add(sentence_vidx_cnt);
				}
				else { // Empty line, document separator
					countTable.add(doc_ct);
					doc_ct = new ArrayList<List<int[]>>();
				}
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		return countTable;
	}
	
	
	public static void writeTopTopicalWords(int[][] M, String fn, int topWords,
											HashMap<Integer, String> id2token)throws IOException
	{
		int K = M.length;
		int V = M[0].length;
		BufferedWriter writer = new BufferedWriter(new FileWriter(fn));
		for(int k=0; k<K; k++) {
			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for(int v=0; v<V; v++) {
				wordCount.put(v, M[k][v]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);
			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for(Integer vIdx : mostLikelyWords){
				if(count<topWords) {
					writer.write(id2token.get(vIdx) + " ");
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
}
