import * as tf from '@tensorflow/tfjs-node';
import * as path from 'node:path';
import * as fs from 'node:fs';
import readLine from 'node:readline';
import {z} from 'zod';
import ora, {type Ora} from 'ora';
import {TFIDFVectorizer} from './TFIDF.ts';

export type documentType = {
	id: number;
	text: string;
};

export const z$dataType = z.object({id: z.number(), text: z.string(), callstack: z.string(), category: z.string()});
export type dataType = z.infer<typeof z$dataType>;

/**
 * Process newline delimited json array.
 */
export class Spinner {
	private title: string = '';
	private index: number = 0;
	private expectedTotal: number | null = null;
	private lastProgress: number = 0;
	private spinner: Ora | null = null;

	public start(title: string, expectedTotal?: number) {
		this.title = title;
		this.index = 0;
		this.expectedTotal = expectedTotal ?? null;
		this.lastProgress = 0;

		if (!this.spinner) {
			this.spinner = ora(title).start();
		}
	}

	public update(index: number) {
		this.index = index;

		if (this.spinner) {
			if (this.expectedTotal !== null) {
				const progress = Math.round((index / this.expectedTotal) * 100);
				if (progress !== this.lastProgress) {
					this.spinner.text = `${this.title} ${index} of ${this.expectedTotal} (${progress}%)`;
					this.spinner.render();
					this.lastProgress = progress;
				}
			} else {
				this.spinner.text = `${this.title} ${index}`;
			}
		}
	}

	public success(title?: string) {
		if (this.spinner) {
			this.spinner.succeed(title ? title : `${this.title} (${this.index + 1})`);
			this.spinner = null;
		}
	}

	public fail(title?: string) {
		if (this.spinner) {
			this.spinner.fail(title ? title : 'Error');
			this.spinner = null;
		}
	}
}

/**
 * Process newline delimited json array.
 */
export const processNewlineDelimitedJsonArray = async (
	filePath: string,
	callback: (value: dataType, index: number) => void,
	limit?: number,
): Promise<number> => {
	const fileStream = fs.createReadStream(filePath, {encoding: 'utf8'});

	const rl = readLine.createInterface({
		input: fileStream,
		crlfDelay: Infinity, // Supports different newline formats
	});

	let index = 0;
	for await (const line of rl) {
		const obj = JSON.parse(line) as unknown;
		const row = z$dataType.parse(obj);

		if (row.category.length > 0) {
			callback(row, index++);
		}

		if (typeof limit === 'number' && index >= limit) {
			break;
		}
	}

	return index;
};

/**
 * Preprocess the data
 */
export const preprocessData = async (filePath: string, maxData?: number): Promise<{vectorizer: TFIDFVectorizer; categories: string[]; totalCount: number}> => {
	const categories: string[] = [];

	// Create and fit the vectorizer
	const vectorizer = new TFIDFVectorizer();

	// Load vocabulary
	const documents: documentType[] = [];
	const totalCount = await processNewlineDelimitedJsonArray(
		filePath,
		(data) => {
			documents.push(data);

			if (!categories.includes(data.category)) {
				categories.push(data.category);
			}
		},
		maxData,
	);

	// Fit vocabulary
	vectorizer.fit(documents);

	return {vectorizer, categories, totalCount};
};

/**
 * Save the trained model
 */
export const saveModel = async (dir: string, model: tf.LayersModel) => {
	const resolvedDir = path.resolve(dir);

	// Ensure the directory exists
	if (!fs.existsSync(resolvedDir)) {
		fs.mkdirSync(resolvedDir, {recursive: true});
	}

	// Save the model
	await model.save(`file://${resolvedDir}`);
};

/**
 * Load the trained model
 */
export const loadModel = async (dir: string): Promise<tf.LayersModel> => {
	const resolvedDir = path.resolve(dir);

	const model = await tf.loadLayersModel(`file://${resolvedDir}/model.json`);

	return model;
};
