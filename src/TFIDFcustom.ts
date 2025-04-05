import {type documentType} from './util.ts';

type Vocabulary = Record<
	string,
	{
		index: number;
		idf: number;
	}
>;

export class TFIDFVectorizer {
	private vocabulary: Vocabulary = {};

	// Step 1: Fit the vectorizer to a corpus of documents
	public fit(documents: documentType[]): void {
		const documentCount = documents.length;
		const documentFrequency: Record<string, number> = {};

		// First pass: build vocabulary and count document frequencies
		for (const doc of documents) {
			// NOTE: cannot use the term "constructor"
			const terms = this.tokenize(doc.text).map((e) => (e === 'constructor' ? 'constructoR' : e));
			const uniqueTerms = new Set(terms);

			for (const term of uniqueTerms) {
				documentFrequency[term] = (documentFrequency[term] || 0) + 1;
			}
		}

		// Second pass: create vocabulary with IDF values
		let index = 0;
		for (const term in documentFrequency) {
			const df = documentFrequency[term];
			const idf = Math.log(documentCount / (1 + df)) + 1; // Smoothed IDF
			this.vocabulary[term] = {index, idf};
			index++;
		}
	}

	public get(text: string): number[] {
		const terms = this.tokenize(text);
		const termFrequency: Record<string, number> = {};
		const vector = new Array<number>(Object.keys(this.vocabulary).length).fill(0);

		// Calculate term frequencies
		for (const term of terms) {
			termFrequency[term] = (termFrequency[term] || 0) + 1;
		}

		// Calculate TF-IDF values
		for (const term in termFrequency) {
			if (this.vocabulary[term]) {
				const tf = termFrequency[term] / terms.length; // Normalized TF
				const idf = this.vocabulary[term].idf;
				const index = this.vocabulary[term].index;
				vector[index] = tf * idf;
			}
		}

		return vector;
	}

	// Helper method to tokenize text (very basic implementation)
	private tokenize(text: string): string[] {
		return (
			text
				.toLowerCase()
				//.replace(/[^\w\s]/g, '') // Remove punctuation
				.split(/\s+/) // Split on whitespace
				.filter((term) => term.length > 0) // Remove empty terms
		);
	}
}
