/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Iterator;

import meka.classifiers.multilabel.BR;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;


/**
 * @author Felipe Bravo
 * usage: java -jar TrainBR.java trainFile.arff testFile.arff testOriginal.csv predictions.csv 
 */
public class TrainBR {

	static public void main(String args[]) throws Exception{

		
		String trainFile=args[0];
		String testFile=args[1];
		String testOriginal=args[2];
		String output=args[3];		
		
		
		// reads train file
		BufferedReader reader = new BufferedReader(
				new FileReader(trainFile));
		Instances train = new Instances(reader);
		reader.close();
		train.setClassIndex(11);
		
		// reads test file
		reader = new BufferedReader(
				new FileReader(testFile));
		Instances test = new Instances(reader);
		reader.close();
		test.setClassIndex(11);
		
		
		// creates multi-label Meka model
		BR mClass=new BR();
		
		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 1 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));
		
		mClass.setClassifier(ll);
		
		
		// trains model
		mClass.buildClassifier(train);
		
		
		// reads original test file 
		reader= new BufferedReader(
				new FileReader(testOriginal));
		
				
		PrintWriter pw=new PrintWriter(output);
		// copies header
		pw.println(reader.readLine());
		
		
		Iterator<Instance> instIt=test.iterator();
		
		while(instIt.hasNext()){
		
			String line=reader.readLine();
			String parts[]=line.split("\t");
			String outLine=parts[0]+"\t"+parts[1]+"\t";
			
			Instance testInst=instIt.next();
			
			
			double[] predictions=mClass.distributionForInstance(testInst);
			
			// convert predictions into desired format
			outLine +=  (int)predictions[0]+"\t"+(int)predictions[1]+"\t"+(int)predictions[2]+"\t"+(int)predictions[3]+"\t"+(int)predictions[4]+"\t"
					+(int)predictions[5]+"\t"+(int)predictions[6]+"\t"+(int)predictions[7]+"\t"+(int)predictions[8]+"\t"
					+(int)predictions[9]+"\t"+(int)predictions[10];
			
			pw.println(outLine);
		
		
			
		}
		
	
		
		
		
		reader.close();
		pw.close();
		
		
	}
	
}
