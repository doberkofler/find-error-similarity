/*
import {type Tensor} from '@tensorflow/tfjs-node';
import {ChartJSNodeCanvas} from 'chartjs-node-canvas';

export const createAndSaveChart = async (history: Record<string, (number | Tensor)[]>): Promise<{lossImage: Buffer; accImage: Buffer | null}> => {
	// Set up Chart.js with Node canvas
	const width = 800;
	const height = 600;
	const backgroundColour = 'rgb(255, 255, 255)';
	const chartJSNodeCanvas = new ChartJSNodeCanvas({width, height, backgroundColour});
	const epochs = Array.from(Array(history.loss.length).keys());

	// Generate loss chart
	if (typeof history.loss !== 'number') {
		throw new Error();
	}
	if (typeof history.val_loss !== 'number') {
		throw new Error();
	}

	const lossImage = await chartJSNodeCanvas.renderToBuffer({
		type: 'line',
		data: {
			labels: epochs,
			datasets: [
				{
					label: 'Training Loss',
					data: history.loss,
					fill: false,
					borderColor: 'rgb(75, 192, 192)',
					tension: 0.1,
				},
				{
					label: 'Validation Loss',
					data: history.val_loss,
					fill: false,
					borderColor: 'rgb(255, 99, 132)',
					tension: 0.1,
				},
			],
		},
		options: {
			plugins: {
				title: {
					display: true,
					text: 'Model Loss During Training',
				},
			},
			scales: {
				x: {
					title: {
						display: true,
						text: 'Epoch',
					},
				},
				y: {
					title: {
						display: true,
						text: 'Loss',
					},
				},
			},
		},
	});

	// Create accuracy chart if available
	if (history.acc) {
		if (typeof history.acc !== 'number') {
			throw new Error();
		}
		if (typeof history.val_acc !== 'number') {
			throw new Error();
		}

		const accImage = await chartJSNodeCanvas.renderToBuffer({
			type: 'line',
			data: {
				labels: epochs,
				datasets: [
					{
						label: 'Training Accuracy',
						data: history.acc,
						fill: false,
						borderColor: 'rgb(75, 192, 192)',
						tension: 0.1,
					},
					{
						label: 'Validation Accuracy',
						data: history.val_acc,
						fill: false,
						borderColor: 'rgb(255, 99, 132)',
						tension: 0.1,
					},
				],
			},
			options: {
				plugins: {
					title: {
						display: true,
						text: 'Model Accuracy During Training',
					},
				},
				scales: {
					x: {
						title: {
							display: true,
							text: 'Epoch',
						},
					},
					y: {
						title: {
							display: true,
							text: 'Accuracy',
						},
						min: 0,
						max: 1,
					},
				},
			},
		});

		return {lossImage, accImage};
	}

	return {lossImage, accImage: null};
};
*/
