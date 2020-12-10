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

namespace NeuralNetwork1
{
    public class NeuralNetwork : BaseNetwork
    {
        private int _input; 
        
        private int _output; 
        
        private Matrix[] _matrices;

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
            _matrices = new Matrix[structure.Count - 1];
            for (var i = 0; i < _matrices.Length; i++)
                _matrices[i] = new Matrix(structure[i], structure[i + 1])
                    .Applied(_ => _random.NextDouble());
        }
        
        public override int Train(Sample sample, bool parallel = true)
        {
            throw new NotImplementedException();
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel = true)
        {
            throw new NotImplementedException();
        }

        public override FigureType Predict(Sample sample)
        {
            throw new NotImplementedException();
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            throw new NotImplementedException();
        }

        public override double[] getOutput()
        {
            throw new NotImplementedException();
        }
    }
}