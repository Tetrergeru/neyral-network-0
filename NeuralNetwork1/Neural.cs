using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.Collections;
using System.Data.SqlTypes;
using System.Windows.Forms.VisualStyles;

namespace NeuralNetwork1
{
    public class NeuralNetwork : BaseNetwork
    {
        private int _input;

        private int _output;

        private Matrix[] _matrices;

        private List<double[]> _results;

        private double[] _result;

        private double _learningRate = 0.25;

        private readonly Random _random = new Random();

        public NeuralNetwork(IReadOnlyList<int> structure)
            => Init(structure, _learningRate);

        public override void ReInit(int[] structure, double initialLearningRate = 0.25)
            => Init(structure, initialLearningRate);

        private void Init(IReadOnlyList<int> structure, double initialLearningRate = 0.25)
        {
            _learningRate = initialLearningRate;
            _input = structure[0];
            _output = structure[structure.Count - 1];
            _result = null;
            _matrices = new Matrix[structure.Count - 1];
            for (var i = 0; i < _matrices.Length; i++)
                _matrices[i] = new Matrix(structure[i], structure[i + 1])
                    .Applied(_ => _random.NextDouble() * 2 - 1);
        }

        public override int Train(Sample sample, bool parallel = true)
        {
            Calculate(sample.input);
            var error = Minus(sample.output, _result);
            var delta = Mult(error, DSigmoid(_results[_matrices.Length]));
            for (var layer = _matrices.Length - 2; layer >= 0; layer--)
            {
                var prevDelta = delta;
                error = delta * _matrices[layer + 1].Transposed();
                _matrices[layer + 1].AddProdFirstTransposed(_results[layer + 1], prevDelta);
                delta = Mult(error, DSigmoid(_results[layer + 1]));
            }
            _matrices[0].AddProdFirstTransposed(_results[0], delta);

            return 0;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError,
            bool parallel = true)
        {
            var start = DateTime.Now;
            for (var i = 0; i < epochsCount; i++)
            {
                foreach (var sample in samplesSet.samples)
                {
                    Train(sample);
                    Console.WriteLine(i);
                }
                
                updateDelegate(
                    (i + 1.0) / epochsCount,
                    0.0, DateTime.Now - start
                );
            }

            return 0.0;
        }

        public override FigureType Predict(Sample sample)
        {
            Calculate(sample.input);
            sample.output = _result;
            sample.ProcessOutput();
            return sample.recognizedClass;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            double correct = 0.0;
            for (var i = 0; i < testSet.Count; ++i)
            {
                Calculate(testSet[i].input);
                testSet[i].output = _result;
                testSet[i].ProcessOutput();
                if (testSet[i].actualClass == testSet[i].recognizedClass) correct += 1;
            }
            return correct/testSet.Count;
        }

        public override double[] getOutput()
        {
            return _result;
        }

        private void Calculate(double[] input)
        {
            _results = new List<double[]>();
            var res = input;
            foreach (var matrix in _matrices)
            {
                _results.Add(res);
                res = Sigmoid(res * matrix);
            }
            _results.Add(res);
            _result = res;
        }

        private static double Sigmoid(double x)
            => 1 / (1 + Math.Exp(-x));

        private static double[] Sigmoid(double[] x)
        {
            var res = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
                res[i] = Sigmoid(x[i]);
            return res;
        }

        private static double DSigmoid(double x)
            => Sigmoid(x) * (1 - Sigmoid(x));
        
        
        private static double[] DSigmoid(double[] x)
        {
            var res = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
                res[i] = DSigmoid(x[i]);
            return res;
        }

        public static double[] Mult(double[] x, double[] y)
        {
            var res = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
                res[i] = x[i] * y[i];
            return res;
        }

        public static double[] Minus(double[] x, double[] y)
        {
            var res = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
                res[i] = x[i] - y[i];
            return res;
        }
    }
}