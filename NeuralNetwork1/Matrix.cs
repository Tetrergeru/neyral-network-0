using System;
using System.Drawing.Printing;
using System.Linq;
using System.Threading;

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

        public static Matrix operator *(Matrix self, double other)
            => self.Applied(v => v * other);
        
        public static Matrix operator *(Matrix self, Matrix other)
            => self.Applied((i, j) =>
                Enumerable.Range(0, self.Width).Select(k => self[i, k] * other[k, j]).Sum());
        
        private Matrix Applied(Func<int, int, double> func)
        {
            var result = new Matrix(Height, Width);
            for (var i = 0; i < result.Height; i++)
            for (var j = 0; j < result.Width; j++)
                result[i, j] = func(i, j);
            return result;
        }
    }
}