import * as tf from '@tensorflow/tfjs-node';
import * as path from 'node:path';
import ora, {type Ora} from 'ora';
import {TFIDFVectorizer} from './TFIDF.ts';
import {processNewlineDelimitedJsonArray, preprocessData, saveModel, Spinner} from './util.ts';

// Convert dataset to tensors
const prepareData = async (
	filePath: string,
	totalCount: number,
	categories: string[],
	vectorizer: TFIDFVectorizer,
	maxLen: number,
	maxData?: number,
): Promise<{features: tf.Tensor2D; labels: tf.Tensor}> => {
	const spinner = new Spinner();

	spinner.start('Prepare vectors to train the model', totalCount);

	const sequences: number[][] = [];
	const categoryIndices: number[] = [];

	await processNewlineDelimitedJsonArray(
		filePath,
		(data, index) => {
			const errorVec = vectorizer.get(data.text).slice(0, maxLen);
			const callstackVec = vectorizer.get(data.callstack).slice(0, maxLen);

			sequences.push([...errorVec, ...callstackVec]);

			const categoryIndex = categories.indexOf(data.category);
			categoryIndices.push(categoryIndex);

			spinner.update(index);
		},
		maxData,
	);

	// Convert to tensors
	const features = tf.tensor2d(sequences);
	const labels = tf.oneHot(tf.tensor1d(categoryIndices, 'int32'), categories.length);

	spinner.success();

	return {features, labels};
};

// For multi-class classification with one-hot encoded labels:
const createModel = (inputSize: number, numCategories: number): tf.LayersModel => {
	const model = tf.sequential();

	// Input layer
	model.add(tf.layers.dense({inputShape: [inputSize * 2], units: 64, activation: 'relu'}));

	//model.add(tf.layers.dense({units: 32, activation: 'relu'}));
	// Hidden layers
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 64, activation: 'relu'}));
	model.add(tf.layers.dense({units: 32, activation: 'relu'}));
	model.add(tf.layers.dense({units: 16, activation: 'relu'}));

	// Output layer with softmax activation
	model.add(tf.layers.dense({units: numCategories, activation: 'softmax'})); // Change to match category count

	model.compile({
		optimizer: 'adam',
		loss: 'categoricalCrossentropy', // Change loss function for multi-class
		metrics: ['accuracy'], // Change metrics to accuracy
	});

	return model;
};

const train = async (features: tf.Tensor2D, labels: tf.Tensor, categories: string[], maxLen: number) => {
	// Create model with correct number of output units
	const model = createModel(maxLen, categories.length);

	const batchSize = 8;
	const validationSplit = 0.2;

	// Calculate total number of batches per epoch
	const totalEpochs = 50;
	let spinnerEpoch: Ora | null = null;
	let currentEpoch = 0;
	const samplesPerEpoch = features.shape[0];
	const totalBatchesPerEpoch = Math.ceil((samplesPerEpoch * (1 - 0.2)) / batchSize); // Adjusting for validation split

	const customCallback = new tf.CustomCallback({
		onEpochBegin: (epoch) => {
			// Start a new spinner for the new epoch
			currentEpoch = epoch;
			spinnerEpoch = ora('').start();
		},
		onBatchEnd: (batch, logs) => {
			if (!spinnerEpoch) {
				return;
			}

			// Calculate progress percentage
			const progress = Math.min(Math.round((batch / totalBatchesPerEpoch) * 100), 100);

			// Only update display on certain intervals to reduce flickering
			spinnerEpoch.text = `Epoch ${currentEpoch + 1}/${totalEpochs} (${progress}%) - Loss: ${logs?.loss?.toFixed(4)}`;
		},
		onEpochEnd: (epoch, logs) => {
			if (spinnerEpoch) {
				spinnerEpoch.succeed(`Epoch ${epoch + 1} completed - Loss: ${logs?.loss?.toFixed(4)}, Val Loss: ${logs?.val_loss?.toFixed(4)}`);
			}
		},
	});

	// Fit
	const history = await model.fit(features, labels, {
		epochs: totalEpochs,
		batchSize,
		validationSplit,
		verbose: 0,
		// The early stopping callback tf.callbacks.earlyStopping({ patience: 5, monitor: 'val_loss' }) will stop training early if the validation loss (val_loss) does not improve for 5 consecutive
		callbacks: [tf.callbacks.earlyStopping({patience: 5, monitor: 'val_loss'}), customCallback],
	});

	const trainedEpochs = history.epoch.length;
	const finalLoss = history.history.loss.slice(-1)[0];
	const finalValLoss = history.history.val_loss?.slice(-1)[0] ?? finalLoss; // Default to loss if val_loss is undefined
	const lossRatio = typeof finalValLoss === 'number' && typeof finalLoss === 'number' ? finalValLoss / finalLoss : null;

	const lossRatioText = lossRatio === null ? '' : ` with a loss ratio of ${lossRatio.toFixed(2)}. (≈ 1.0 → Good fit, > 1.2 → Overfitting, < 0.8 → Underfitting`;
	const spinner = ora('Train').start();
	spinner.succeed(`Trained completed after ${trainedEpochs} epochs${lossRatioText}`);

	return {model, history};
};

// Main function
const main = async () => {
	const dir = 'data';
	const filePath = path.join(dir, 'traning_success.json');
	const maxLen = 50;
	const maxData = undefined;

	// First, we preprocess the data
	const {vectorizer, categories, totalCount} = await preprocessData(filePath, maxData);

	// Prepare data
	const {features, labels} = await prepareData(filePath, totalCount, categories, vectorizer, maxLen, maxData);

	// Train model
	const {model /*, history*/} = await train(features, labels, categories, maxLen);

	// Create chart
	/*
	const charts = await createAndSaveChart(history.history);
	if (charts) {
		const {lossImage, accImage} = charts;

		writeFileBuffer('loss_chart.png', lossImage);
		console.log('Loss chart saved as "loss_chart.png".');
		if (accImage) {
			writeFileBuffer('accuracy_chart.png', accImage);
			console.log('Accuracy chart saved as "accuracy_chart.png".');
		}
	}
	*/

	// Svae model
	await saveModel(dir, model);
};

// Run the training process
main().catch(console.error);
