import natural from 'natural';
import {type documentType, Spinner} from './util.ts';

type VocabularyType = {
	terms: string[];
	idf: Record<string, number>;
	documents: Record<number, Record<string, number>>;
	termIndices: Record<string, number>; // For O(1) lookups
};

export class TFIDFVectorizer {
	private vocabulary: VocabularyType = {
		terms: [],
		idf: {},
		documents: {},
		termIndices: {},
	};
	private tokenizer: natural.WordTokenizer;

	constructor() {
		this.tokenizer = new natural.WordTokenizer();
	}

	// Create a vocabulary from a corpus of documents
	public fit(documents: documentType[]): void {
		const spinner = new Spinner();

		// Reset the TF-IDF instance and vocabulary
		const tfidf = new natural.TfIdf();
		this.vocabulary = {
			terms: [],
			idf: {},
			documents: {},
			termIndices: {},
		};

		// Add all documents to the TF-IDF calculator
		spinner.start('Add all documents to the TF-IDF calculator', documents.length);
		let index = 0;
		for (const doc of documents) {
			tfidf.addDocument(doc.text);

			// Spinner
			spinner.update(index);

			index++;
		}
		spinner.success();

		// Extract all unique terms and build vocabulary
		spinner.start('Extract all unique terms and build vocabulary', documents.length);
		const termSet = new Set<string>();
		index = 0;
		for (const doc of documents) {
			// Store document terms with their TF-IDF values
			const docTerms: Record<string, number> = {};

			// Get all terms in the document
			const tokens = this.tokenizer.tokenize(doc.text.toLowerCase()) || [];

			tokens.forEach((term) => {
				termSet.add(term);

				// Calculate TF-IDF for this term in this document
				const tfidfValue = tfidf.tfidf(term, index);
				docTerms[term] = tfidfValue;
			});

			// Store document's term vectors
			this.vocabulary.documents[doc.id] = docTerms;

			// Spinner
			spinner.update(index);

			index++;
		}
		spinner.success();

		// Convert the set to an array and sort alphabetically
		this.vocabulary.terms = Array.from(termSet).sort();

		// Build term indices for O(1) lookups
		this.vocabulary.terms.forEach((term, i) => {
			this.vocabulary.termIndices[term] = i;
		});

		// Calculate and store IDF values for each term
		spinner.start('Calculate and store IDF values for each term', this.vocabulary.terms.length);
		index = 0;
		for (const term of this.vocabulary.terms) {
			// Use the TfIdf instance to get IDF value for each term
			// First, we need to find the IDF by dividing a term's TFIDF by its TF in a document where it exists
			let termIdf = 0;

			// Find a document that contains this term
			for (let i = 0; i < tfidf.documents.length; i++) {
				const doc = tfidf.documents[i];
				const tf = doc[term] || 0;

				if (tf > 0) {
					// Get tfidf value
					const tfidfValue = tfidf.tfidf(term, i);
					// Calculate IDF by dividing tfidf by tf
					termIdf = tfidfValue / tf;
					break;
				}
			}

			this.vocabulary.idf[term] = termIdf;

			// Spinner
			spinner.update(index);

			index++;
		}
		spinner.success();
	}

	// Get the vector for a given text
	public get(text: string): number[] {
		if (this.vocabulary.terms.length === 0) {
			throw new Error('Vectorizer must be fitted before getting vectors');
		}

		// Initialize result vector with zeros
		const result = new Array(this.vocabulary.terms.length).fill(0);

		// Calculate term frequencies directly
		const tokens = this.tokenizer.tokenize(text.toLowerCase()) || [];
		const totalTerms = tokens.length;

		if (totalTerms === 0) {
			return result; // Return all zeros for empty text
		}

		// First pass: count occurrences of terms
		const counts: Record<string, number> = {};
		for (const token of tokens) {
			counts[token] = (counts[token] || 0) + 1;
		}

		// Second pass: calculate TF-IDF for terms that exist in our vocabulary
		for (const term in counts) {
			const index = this.vocabulary.termIndices[term];
			if (index !== undefined) {
				// Only process terms in our vocabulary
				const tf = counts[term] / totalTerms;
				const idf = this.vocabulary.idf[term] || 0;
				result[index] = tf * idf;
			}
		}

		return result;
	}
}
