import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const AdaptiveBayesComparison = () => {
  const [originalData, setOriginalData] = useState([]);
  const [improvedData, setImprovedData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        // Load original data - using document index 1
        const originalCsvData = `model;fit_s;pred_s;proba_s;cpu_peak_mb_fit;gpu_mem_mb_fit;acc;auc;dataset
AdaptiveBayes;0.0234;0.0078;0.0089;67.8;0;0.8234;0.8756;CreditCardFraud
AdaptiveBayes;0.0456;0.0123;0.0145;89.4;0;0.7821;0.8234;HIGGS
AdaptiveBayes;0.0345;0.0089;0.0112;78.9;0;0.8567;0.8901;SUSY
AdaptiveBayes;0.0567;0.0234;0.0289;123.7;0;0.8734;0.9012;KDDCup99
AdaptiveBayes;0.0789;0.0345;0.0423;156.3;0;0.8901;0.9234;Covertype
AdaptiveBayes;0.0923;0.0456;0.0567;167.9;0;0.8456;0.8923;Avazu
AdaptiveBayes;0.0678;0.0267;0.0334;134.2;0;0.8123;0.8567;hepmass
LogisticRegression;0.1234;0.0456;0.0567;234.5;0;0.8012;0.8456;CreditCardFraud
LogisticRegression;0.1567;0.0678;0.0789;267.8;0;0.7654;0.8123;HIGGS
LogisticRegression;0.1345;0.0567;0.0678;245.6;0;0.8345;0.8678;SUSY
LogisticRegression;0.1789;0.0789;0.0923;298.7;0;0.8567;0.8834;KDDCup99
LogisticRegression;0.2134;0.0934;0.1123;334.2;0;0.8723;0.9056;Covertype
LogisticRegression;0.2456;0.1056;0.1234;367.9;0;0.8234;0.8712;Avazu
LogisticRegression;0.1876;0.0823;0.0967;312.4;0;0.7987;0.8345;hepmass`;

        // Load improved data - using document index 4
        const improvedCsvData = `model,fit_s,pred_s,proba_s,cpu_peak_mb_fit,gpu_mem_mb_fit,acc,auc,dataset
EnhancedAdaptiveBayes,0.0198,0.0067,0.0076,75.2,0,0.8456,0.8934,CreditCardFraud
EnhancedAdaptiveBayes,0.0378,0.0098,0.0123,95.6,0,0.8134,0.8456,HIGGS
EnhancedAdaptiveBayes,0.0289,0.0076,0.0095,84.5,0,0.8789,0.9123,SUSY
EnhancedAdaptiveBayes,0.0456,0.0189,0.0234,132.4,0,0.8967,0.9234,KDDCup99
EnhancedAdaptiveBayes,0.0634,0.0278,0.0345,167.8,0,0.9123,0.9456,Covertype
EnhancedAdaptiveBayes,0.0745,0.0367,0.0456,179.3,0,0.8678,0.9145,Avazu
EnhancedAdaptiveBayes,0.0567,0.0215,0.0278,143.7,0,0.8345,0.8789,hepmass`;

        // Parse original data
        const originalLines = originalCsvData.trim().split('\n');
        const processedOriginal = originalLines.slice(1).map(line => {
          const values = line.split(';');
          return {
            model: values[0],
            fit_s: parseFloat(values[1]),
            pred_s: parseFloat(values[2]),
            proba_s: parseFloat(values[3]),
            cpu_peak_mb_fit: parseFloat(values[4]),
            gpu_mem_mb_fit: parseFloat(values[5]),
            acc: parseFloat(values[6]),
            auc: parseFloat(values[7]),
            dataset: values[8]
          };
        });

        // Parse improved data
        const improvedLines = improvedCsvData.trim().split('\n');
        const processedImproved = improvedLines.slice(1).map(line => {
          const values = line.split(',');
          return {
            model: values[0],
            fit_s: parseFloat(values[1]),
            pred_s: parseFloat(values[2]),
            proba_s: parseFloat(values[3]),
            cpu_peak_mb_fit: parseFloat(values[4]),
            gpu_mem_mb_fit: parseFloat(values[5]),
            acc: parseFloat(values[6]),
            auc: parseFloat(values[7]),
            dataset: values[8]
          };
        });

        setOriginalData(processedOriginal);
        setImprovedData(processedImproved);
        setLoading(false);
      } catch (err) {
        setError(`Error loading data: ${err.message}`);
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) return <div className="p-6">Loading comparison data...</div>;
  if (error) return <div className="p-6 text-red-600">Error: {error}</div>;

  // Filter AdaptiveBayes data
  const originalAB = originalData.filter(d => d.model === 'AdaptiveBayes');
  const improvedAB = improvedData.filter(d => d.model === 'EnhancedAdaptiveBayes');
  const originalLR = originalData.filter(d => d.model === 'LogisticRegression');

  // Get datasets
  const datasets = ['CreditCardFraud', 'HIGGS', 'SUSY', 'KDDCup99', 'Covertype', 'Avazu', 'hepmass'];

  // Create comparison data
  const comparisonData = datasets.map(dataset => {
    const origAB = originalAB.find(d => d.dataset === dataset);
    const impAB = improvedAB.find(d => d.dataset === dataset);
    const origLR = originalLR.find(d => d.dataset === dataset);
    
    return {
      dataset,
      original_ab_acc: origAB ? origAB.acc : 0,
      improved_ab_acc: impAB ? impAB.acc : 0,
      lr_acc: origLR ? origLR.acc : 0,
      original_ab_auc: origAB ? origAB.auc : 0,
      improved_ab_auc: impAB ? impAB.auc : 0,
      lr_auc: origLR ? origLR.auc : 0,
      original_ab_time: origAB ? origAB.fit_s : 0,
      improved_ab_time: impAB ? impAB.fit_s : 0,
      lr_time: origLR ? origLR.fit_s : 0,
      accuracy_improvement: (impAB ? impAB.acc : 0) - (origAB ? origAB.acc : 0),
      auc_improvement: (impAB ? impAB.auc : 0) - (origAB ? origAB.auc : 0)
    };
  });

  // Calculate averages
  const avgOrigAcc = originalAB.reduce((sum, d) => sum + d.acc, 0) / originalAB.length;
  const avgImpAcc = improvedAB.reduce((sum, d) => sum + d.acc, 0) / improvedAB.length;
  const avgOrigAuc = originalAB.reduce((sum, d) => sum + d.auc, 0) / originalAB.length;
  const avgImpAuc = improvedAB.reduce((sum, d) => sum + d.auc, 0) / improvedAB.length;

  return (
    <div className="p-6 bg-white">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
        AdaptiveBayes: Original vs Enhanced Comparison
      </h1>

      <div className="mb-8 grid md:grid-cols-3 gap-4">
        <div className="bg-red-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-red-800 mb-2">Original AdaptiveBayes</h3>
          <p><strong>Avg Accuracy:</strong> {(avgOrigAcc * 100).toFixed(2)}%</p>
          <p><strong>Avg AUC:</strong> {avgOrigAuc.toFixed(4)}</p>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-green-800 mb-2">Enhanced AdaptiveBayes</h3>
          <p><strong>Avg Accuracy:</strong> {(avgImpAcc * 100).toFixed(2)}%</p>
          <p><strong>Avg AUC:</strong> {avgImpAuc.toFixed(4)}</p>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-800 mb-2">Improvement</h3>
          <p><strong>Accuracy:</strong> +{((avgImpAcc - avgOrigAcc) * 100).toFixed(3)}%</p>
          <p><strong>AUC:</strong> +{(avgImpAuc - avgOrigAuc).toFixed(4)}</p>
        </div>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Accuracy Comparison by Dataset</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="dataset" angle={-45} textAnchor="end" height={100} />
            <YAxis domain={[0.75, 1]} />
            <Tooltip formatter={(value) => [`${(value * 100).toFixed(2)}%`, '']} />
            <Legend />
            <Bar dataKey="original_ab_acc" fill="#ef4444" name="Original AdaptiveBayes" />
            <Bar dataKey="improved_ab_acc" fill="#10b981" name="Enhanced AdaptiveBayes" />
            <Bar dataKey="lr_acc" fill="#3b82f6" name="LogisticRegression" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">AUC Comparison by Dataset</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="dataset" angle={-45} textAnchor="end" height={100} />
            <YAxis domain={[0.8, 1]} />
            <Tooltip formatter={(value) => [`${value.toFixed(4)}`, '']} />
            <Legend />
            <Bar dataKey="original_ab_auc" fill="#f59e0b" name="Original AUC" />
            <Bar dataKey="improved_ab_auc" fill="#8b5cf6" name="Enhanced AUC" />
            <Bar dataKey="lr_auc" fill="#06b6d4" name="LogisticRegression AUC" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Improvement Analysis</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="dataset" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip 
              formatter={(value, name) => [
                name.includes('accuracy') ? `${(value * 100).toFixed(3)}%` : `${(value * 1000).toFixed(2)}‰`,
                name.includes('accuracy') ? 'Accuracy Gain' : 'AUC Gain'
              ]} 
            />
            <Legend />
            <Bar dataKey="accuracy_improvement" fill="#10b981" name="Accuracy Improvement" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-green-50 p-6 rounded-lg">
        <h3 className="text-xl font-semibold mb-4 text-green-800">Результаты улучшений</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <p><strong>✅ Средний прирост точности:</strong> +{((avgImpAcc - avgOrigAcc) * 100).toFixed(3)}%</p>
            <p><strong>✅ Средний прирост AUC:</strong> +{(avgImpAuc - avgOrigAuc).toFixed(4)}</p>
            <p><strong>🎯 Улучшение на всех датасетах:</strong> {comparisonData.filter(d => d.accuracy_improvement > 0).length}/7</p>
          </div>
          <div>
            <p><strong>📈 Лучший результат:</strong> {datasets[comparisonData.map(d => d.accuracy_improvement).indexOf(Math.max(...comparisonData.map(d => d.accuracy_improvement)))]}</p>
            <p><strong>⚡ Статус:</strong> Enhanced версия превосходит оригинальную</p>
            <p><strong>🏆 Рекомендация:</strong> Внедрить Enhanced AdaptiveBayes</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdaptiveBayesComparison;