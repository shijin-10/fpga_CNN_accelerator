module conv1_mnist_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] weights [0:71];
    reg signed [31:0] biases [0:7];

    wire signed [31:0] output_data [0:6271];

    integer i, f;

    conv1_8filters_bias dut (
        .image(image),
        .weights(weights),
        .biases(biases),
        .output_data(output_data)
    );

    initial begin
        $readmemh("mnist_image.mem", image);
        $readmemh("conv1_weights.mem", weights);
        $readmemh("conv1_bias_int32.mem", biases);

        #10;

        $display("MNIST image: label 7");
        $display("Conv1 + Bias + ReLU");
        $display("First pooled-position candidates:");

        for (f = 0; f < 8; f = f + 1)
            $display("Filter %0d, Output[0] = %0d",
                     f, output_data[f*784]);

        $finish;
    end

endmodule
