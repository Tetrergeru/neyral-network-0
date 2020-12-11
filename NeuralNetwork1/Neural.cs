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

        private List<Matrix> _results;

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
                    .Applied(_ => _random.NextDouble());
        }

        public override int Train(Sample sample, bool parallel = true)
        {
            Calculate(sample.input);
            sample.output = _result;
            sample.ProcessOutput();
            var error = Matrix.Row(sample.error);
            var delta = error ^ _results[_matrices.Length].Applied(DSigmoid);
            _matrices[_matrices.Length - 1] += _results[_matrices.Length - 1].Transposed() * delta;
            for (var layer = _matrices.Length - 2; layer >= 0; layer--)
            {
                error = delta * _matrices[layer + 1].Transposed();
                delta = error ^ _results[layer + 1].Applied(DSigmoid);
                _matrices[layer] += _results[layer].Transposed() * delta;
            }
            return 0;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError,
            bool parallel = true)
        {
            var start = DateTime.Now;
            for (var i = 0; i < epochsCount; i++)
            {
                updateDelegate((i + 1.0) / epochsCount, 0.0, DateTime.Now - start);
                foreach (var sample in samplesSet.samples)
                {
                    Train(sample);
                    Console.WriteLine(i);
                }
            }

            return 0.0;
        }

        public override FigureType Predict(Sample sample)
        {
            Calculate(sample.input);
            sample.output = _result;
            return FigureType.Undef;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            foreach (var sample in testSet.samples)
                Calculate(sample.input);
            return 0.0;
        }

        public override double[] getOutput()
        {
            return _result;
        }

        private void Calculate(IReadOnlyList<double> input)
        {
            _results = new List<Matrix>();
            var res = Matrix.Row(input);
            foreach (var matrix in _matrices)
            {
                _results.Add(res);
                res = (res * matrix).Applied(Sigmoid);
            }
            _results.Add(res);
            _result = res.ToRow().ToArray();
        }

        private static double Sigmoid(double x)
            => 1 / (1 + Math.Exp(-x));

        private static double DSigmoid(double x)
            => Sigmoid(x) * (1 - Sigmoid(x));
    }
}