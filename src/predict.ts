import * as path from 'node:path';
import * as tf from '@tensorflow/tfjs-node';
import {TFIDFVectorizer} from './TFIDF.ts';
import {preprocessData, processNewlineDelimitedJsonArray, loadModel, Spinner} from './util.ts';

export class ErrorClassifier {
	private model: tf.LayersModel;
	private vectorizer: TFIDFVectorizer;
	private categories: string[];
	private maxLen: number;

	constructor(model: tf.LayersModel, vectorizer: TFIDFVectorizer, categories: string[], maxLen: number) {
		this.model = model;
		this.vectorizer = vectorizer;
		this.categories = categories;
		this.maxLen = maxLen;
	}

	/**
	 * Predict the category for a single error
	 */
	predict(
		text: string,
		callstack: string,
	): {
		category: string;
		confidence: number;
		allPredictions: {category: string; confidence: number}[];
	} {
		if (!this.model || !this.vectorizer || this.categories.length === 0) {
			throw new Error('Model, vectorizer, or categories not loaded');
		}

		// Transform text and callstack to vectors
		const textVec = this.vectorizer.get(text).slice(0, this.maxLen);
		const callstackVec = this.vectorizer.get(callstack).slice(0, this.maxLen);

		// Combine vectors
		const features = [...textVec, ...callstackVec];

		// Create tensor and reshape for the model
		const inputTensor = tf.tensor2d([features]);

		// Make prediction
		const predictions = this.model.predict(inputTensor) as tf.Tensor;
		const predictionData = predictions.dataSync();

		// Get highest confidence prediction
		const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
		const predictedCategory = this.categories[maxIndex];
		const confidence = predictionData[maxIndex];

		// Get all category predictions with confidence scores
		const allPredictions = this.categories
			.map((category, index) => ({
				category,
				confidence: predictionData[index],
			}))
			.sort((a, b) => b.confidence - a.confidence);

		// Clean up tensors
		inputTensor.dispose();
		predictions.dispose();

		return {
			category: predictedCategory,
			confidence,
			allPredictions,
		};
	}
}

const main = async () => {
	const dir = 'data';
	const filePath = path.join(dir, 'traning_success.json');
	const maxLen = 50;
	const maxData = undefined;
	const spinner = new Spinner();

	try {
		// First, we preprocess the data
		const {vectorizer, categories, totalCount} = await preprocessData(filePath, maxData);

		// Load model
		const model = await loadModel(dir);

		// Create classifier
		const classifier = new ErrorClassifier(model, vectorizer, categories, maxLen);

		spinner.start('Predict', totalCount);

		let successCount = 0;
		let failureCount = 0;
		await processNewlineDelimitedJsonArray('./data/traning_success.json', (data, index) => {
			// Make prediction
			const result = classifier.predict(data.text, data.callstack);

			//console.log('Predicted category:', result.category);
			//console.log('Confidence:', result.confidence);
			//console.log('All predictions:', result.allPredictions);

			spinner.update(index);

			if (result.category === data.category) {
				successCount++;
			} else {
				failureCount++;
			}
		});

		spinner.success(`${successCount} successes and ${failureCount} failures.`);
	} catch (e: unknown) {
		spinner.fail();
		console.log(e);
	}
};

// Run prediction example
main().catch(console.error);
