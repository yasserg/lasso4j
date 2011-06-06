package edu.uci.lasso;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class TestLasso {

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new InputStreamReader(TestLasso.class.getClassLoader().getResourceAsStream("diabetes.data")));
		String line = reader.readLine(); // ignore header
		String[] parts = line.split("\t");
		int featuresCount = parts.length - 1;
		List<float[]> observations = new ArrayList<float[]>();
		List<Float> targets = new ArrayList<Float>();
		while ((line = reader.readLine()) != null) {
			parts = line.split("\t");
			float[] curObservation = new float[featuresCount];
			for (int f = 0; f < featuresCount; f++) {
				curObservation[f] = Float.parseFloat(parts[f]);
			}
			observations.add(curObservation);
			targets.add(Float.parseFloat(parts[parts.length - 1]));
		}
		
		LassoFitGenerator fitGenerator = new LassoFitGenerator();
		int numObservations = observations.size();
		fitGenerator.init(featuresCount, numObservations);
		for (int i = 0; i < numObservations; i++) {
			fitGenerator.setObservationValues(i, observations.get(i));
			fitGenerator.setTarget(i, targets.get(i));
		}
		fitGenerator.fit(-1);
	}
}
