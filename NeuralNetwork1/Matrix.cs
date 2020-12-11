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

        public IReadOnlyList<double> ToRow()
            => Height > 1 ? throw new ArgumentException("Matrix should be one row to use ToRow") : _matrix;

        public IReadOnlyList<double> ToCol()
            => Width > 1 ? throw new ArgumentException("Matrix should be one col to use ToCol") : _matrix;

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
            return self.Applied((i, j) => self[i, j] + other[i, j]);
        }

        public static Matrix operator ^(Matrix self, Matrix other)
        {
            if (self.Width != other.Width || self.Height != other.Height)
                throw new ArgumentException($"self.Width != other.Width || self.Height != other.Height");
            return self.Applied((i, j) => self[i, j] * other[i, j]);
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

        public static Matrix Column(IReadOnlyList<double> column)
            => new Matrix(column.Count, 1).Applied((i, _) => column[i]);
        
        public static Matrix Row(IReadOnlyList<double> row)
            => new Matrix(1, row.Count).Applied((_, j) => row[j]);
    }
}