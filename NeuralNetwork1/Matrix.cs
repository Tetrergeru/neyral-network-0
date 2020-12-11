using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    /*    Width
     * *<=======> J
     * ^
     * |
     * | Height
     * |
     * I
     */
    public class Matrix
    {
        private readonly double[] _matrix;

        public int Width { get; }

        public int Height { get; }

        public Matrix(int height, int width)
        {
            Width = width;
            Height = height;
            _matrix = new double[width * height];
        }

        public double this[int i, int j]
        {
            get => _matrix[i * Width + j];
            set => _matrix[i * Width + j] = value;
        }

        public Matrix Applied(Func<double, double> func)
            => Applied((i, j) => func(this[i, j]));

        public Matrix Transposed()
            => new Matrix(Width, Height).Apply((i, j) => this[j, i]);

        public static Matrix operator *(Matrix self, double other)
            => self.Applied(v => v * other);

        public static Matrix operator +(Matrix self, Matrix other)
        {
            if (self.Width != other.Width || self.Height != other.Height)
                throw new ArgumentException($"self.Width != other.Width || self.Height != other.Height");
            
            var result = new Matrix(self.Height, self.Width);
            foreach (var i in Enumerable.Range(0, result.Height))
                Parallel.For(0, result.Width, j => { result[i, j] = self[i, j] + other[i, j]; });
            return result;
        }

        public static Matrix operator ^(Matrix self, Matrix other)
        {
            if (self.Width != other.Width || self.Height != other.Height)
                throw new ArgumentException($"self.Width != other.Width || self.Height != other.Height");
            var result = new Matrix(self.Height, self.Width);
            foreach (var i in Enumerable.Range(0, result.Height))
                Parallel.For(0, result.Width, j => { result[i, j] = self[i, j] * other[i, j]; });
            return result;
        }

        public static Matrix operator *(Matrix self, Matrix other)
        {
            if (self.Width != other.Height)
                throw new ArgumentException($"self.Width != other.Height; {self.Width} != {other.Height}");
            var res = new Matrix(self.Height, other.Width).Apply((i, j) =>
                Enumerable.Range(0, self.Width).Select(k => self[i, k] * other[k, j]).Sum());
            return res;
        }

        private Matrix Applied(Func<int, int, double> func)
        {
            var result = new Matrix(Height, Width);
            foreach (var i in Enumerable.Range(0, Height))
                Parallel.For(0, result.Width, j => { result[i, j] = func(i, j); });
            return result;
        }

        private Matrix Apply(Func<int, int, double> func)
        {
            foreach (var i in Enumerable.Range(0, Height))
                Parallel.For(0, this.Width, j => { this[i, j] = func(i, j); });
            return this;
        }

        public static double[] operator *(IReadOnlyList<double> self, Matrix other)
        {
            if (self.Count != other.Height)
                throw new ArgumentException($"self.Width != other.Height; {self.Count} != {other.Height}");
            var res = new double[other.Width];
            Parallel.For(0, res.Length, i => { 
                res[i] = self.Select((t, k) => t * other[k, i]).Sum();
            });

            return res;
        }

        public void AddProdFirstTransposed(double[] first, double[] second)
        {
            for (var i = 0; i < Height; i++)
            for (var j = 0; j < Width; j++)
                this[i, j] += first[i] * second[j];
        }
    }
}