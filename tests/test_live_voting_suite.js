import { detectWithEnsembleVoting } from '../src/services/predictionApi.js';
import fs from 'fs';
import path from 'path';

async function main() {
  const testImages = [
    'test_images/ai-generated.png',
    'test_images/ai_retouched.png',
    'test_images/human.jpeg',
    'public/images/sample-fake-woman.jpg',
    'public/images/sample-real-man.jpg'
  ];

  console.log('=== LIVE EVALUATION: 3 DETECTORS & ENSEMBLE VOTING ===\n');

  for (const imgPath of testImages) {
    if (!fs.existsSync(imgPath)) continue;
    const buf = fs.readFileSync(imgPath);
    const blob = new Blob([buf], { type: 'image/jpeg' });
    blob.name = path.basename(imgPath);

    const res = await detectWithEnsembleVoting(blob);
    console.log('----------------------------------------------------');
    console.log('Image:', path.basename(imgPath));
    console.log('Final Voting Decision:', res.prediction, '| Confidence:', (res.confidence * 100).toFixed(1) + '%', '| Votes:', JSON.stringify(res.vote_counts));
    console.log('Individual Detectors:');
    for (const d of res.individual_detectors) {
      const pAiStr = d.probability_ai !== null ? (d.probability_ai * 100).toFixed(1) + '%' : 'N/A';
      const pRealStr = d.probability_real !== null ? (d.probability_real * 100).toFixed(1) + '%' : 'N/A';
      console.log('  -', d.name + ':', d.label, `(P(AI)=${pAiStr}, P(Real)=${pRealStr})`);
    }
  }
}

main().catch(console.error);
