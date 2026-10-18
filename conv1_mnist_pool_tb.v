
module conv1_mnist_pool_tb;

    reg signed [7:0] image [0:783];
    reg signed [7:0] weights [0:71];
    reg signed [31:0] biases [0:7];

    wire signed [31:0] conv_output [0:6271];
    wire signed [31:0] pool_output [0:1567];

    integer f;
    integer outfile;

    conv1_8filters_bias conv1 (
        .image(image),
        .weights(weights),
        .biases(biases),
        .output_data(conv_output)
    );

    maxpool_2x2 pool (
        .input_data(conv_output),
        .output_data(pool_output)
    );

    initial begin
        $readmemh("mnist_image.mem", image);
        $readmemh("conv1_weights.mem", weights);
        $readmemh("conv1_bias_int32.mem", biases);

        #10;

        outfile = $fopen("conv1_pool_output.mem", "w");

        for (f = 0; f < 1568; f = f + 1)
            $fdisplay(outfile, "%0d", pool_output[f]);

        $fclose(outfile);

        $display("Saved 1568 pooled outputs to conv1_pool_output.mem");
        $finish;
    end

endmodule

