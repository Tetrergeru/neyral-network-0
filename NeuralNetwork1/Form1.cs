﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{

	public delegate void FormUpdater(double progress, double error, TimeSpan time);

    public partial class Form1 : Form
    {
        /// <summary>
        /// Генератор изображений (образов)
        /// </summary>
        GenerateImage generator = new GenerateImage();
        
        /// <summary>
        /// Самодельный персептрон – из массивов и палок
        /// </summary>
        NeuralNetwork CustomNet = null;

        /// <summary>
        /// Обёртка для ActivationNetwork из Accord.Net
        /// </summary>
        AccordNet AccordNet = null;

        /// <summary>
        /// Абстрактный базовый класс, псевдоним либо для CustomNet, либо для AccordNet
        /// </summary>
        BaseNetwork net = null;

        public Form1()
        {
            InitializeComponent();
            netTypeBox.SelectedIndex = 1;
            generator.figure_count = (int)classCounter.Value;
            button3_Click(this, null);
            pictureBox1.Image = Properties.Resources.Title;
        }

		public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
		{
			if (progressBar1.InvokeRequired)
			{
				progressBar1.Invoke(new FormUpdater(UpdateLearningInfo),new Object[] {progress, error, elapsedTime});
				return;
			}

            StatusLabel.Text = "Accuracy: " + error;
            int prgs = (int)Math.Round(progress*100);
			prgs = Math.Min(100, Math.Max(0,prgs));
            elapsedTimeLabel.Text = "Затраченное время : " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = prgs;
		}


        private void set_result(Sample figure)
        {
            label1.Text = figure.ToString();

            label1.ForeColor = figure.Correct() ? Color.Green : Color.Red;

            label1.Text = "Распознано : " + figure.recognizedClass.ToString();

            label8.Text = string.Join("\n", net.getOutput());
            pictureBox1.Image = generator.genBitmap();
            pictureBox1.Invalidate();
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            Sample fig = generator.GenerateFigure();

            net.Predict(fig);

            set_result(fig);
        }

        private async Task<double> train_networkAsync(int training_size, int epoches, double acceptable_error, bool parallel = true)
        {
            //  Выключаем всё ненужное
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            trainOneButton.Enabled = false;

            //  Создаём новую обучающую выборку
            SamplesSet samples = new SamplesSet();

            for (int i = 0; i < training_size; i++)
                samples.AddSample(generator.GenerateFigure());

            //  Обучение запускаем асинхронно, чтобы не блокировать форму
            double f = await Task.Run(() => net.TrainOnDataSet(samples, epoches, acceptable_error, parallel));

            label1.Text = "Щелкните на картинку для теста нового образа";
            label1.ForeColor = Color.Green;
            groupBox1.Enabled = true;
            pictureBox1.Enabled = true;
            trainOneButton.Enabled = true;
            StatusLabel.Text = "Accuracy: " + f.ToString();
            StatusLabel.ForeColor = Color.Green;
            return f;

        }

        private void button1_Click(object sender, EventArgs e)
        {

            #pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
            train_networkAsync( (int)TrainingSizeCounter.Value, (int)EpochesCounter.Value, (100 - AccuracyCounter.Value) / 100.0, parallelCheckBox.Checked);
            #pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        }

        private void button2_Click(object sender, EventArgs e)
        {
            this.Enabled = false;
            //  Тут просто тестирование новой выборки
            //  Создаём новую обучающую выборку
            var samples = new SamplesSet();

            for (var i = 0; i < (int)TrainingSizeCounter.Value; i++)
                samples.AddSample(generator.GenerateFigure());

            var accuracy = net.TestOnDataSet(samples);
            
            StatusLabel.Text = string.Format("Точность на тестовой выборке : {0,5:F2}%", accuracy*100);
            StatusLabel.ForeColor = accuracy*100 >= AccuracyCounter.Value ? Color.Green : Color.Red;

            Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            //  Проверяем корректность задания структуры сети
            var structure = netStructureBox.Text.Split(';').Select((c) => int.Parse(c)).ToArray();
            if (structure.Length < 2 || structure[0] != 400 || structure[structure.Length - 1] != generator.figure_count)
            {
                MessageBox.Show("А давайте вы структуру сети нормально запишите, ОК?", "Ошибка", MessageBoxButtons.OK);
                return;
            };

            CustomNet = new NeuralNetwork(structure) {updateDelegate = UpdateLearningInfo};

            AccordNet = new AccordNet(structure) {updateDelegate = UpdateLearningInfo};

            SetNetwork();
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            generator.figure_count = (int)classCounter.Value;
            var vals = netStructureBox.Text.Split(';');
            int outputNeurons;
            if (int.TryParse(vals.Last(), out outputNeurons))
            {
                vals[vals.Length - 1] = classCounter.Value.ToString();
                netStructureBox.Text = vals.Aggregate((partialPhrase, word) => $"{partialPhrase};{word}");
            }
        }

        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            if (net == null) return;
            Sample fig = generator.GenerateFigure();
            pictureBox1.Image = generator.genBitmap();
            pictureBox1.Invalidate();
            net.Train(fig, false);
            set_result(fig);
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void testNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Тестировать нейросеть на тестовой выборке такого же размера";
        }

        private void netTypeBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            SetNetwork();
        }

        private void SetNetwork()
        {
            if (netTypeBox.SelectedIndex == 0)
                net = CustomNet;
            else
                net = AccordNet;
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Заново пересоздаёт сеть с указанными параметрами";
        }
    }

  }
