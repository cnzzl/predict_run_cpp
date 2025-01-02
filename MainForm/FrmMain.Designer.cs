namespace MainForm
{
    partial class FrmMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            btnONNX2TRT = new Button();
            SuspendLayout();
            // 
            // btnONNX2TRT
            // 
            btnONNX2TRT.Location = new Point(603, 152);
            btnONNX2TRT.Name = "btnONNX2TRT";
            btnONNX2TRT.Size = new Size(80, 31);
            btnONNX2TRT.TabIndex = 0;
            btnONNX2TRT.Text = "onnx转trt";
            btnONNX2TRT.UseVisualStyleBackColor = true;
            btnONNX2TRT.Click += btnONNX2TRT_Click;
            // 
            // FrmMain
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(btnONNX2TRT);
            Name = "FrmMain";
            Text = "Form1";
            Load += Form1_Load;
            ResumeLayout(false);
        }

        #endregion

        private Button btnONNX2TRT;
    }
}