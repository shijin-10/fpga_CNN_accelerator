
module conv2_mnist_tb;

    reg signed [7:0] input_data [0:1567];
    reg signed [7:0] weights [0:1151];
    reg signed [31:0] biases [0:15];

    wire signed [31:0] output_data [0:3135];

    integer i;
    integer outfile;

    conv2_16filters dut (
        .input_data(input_data),
        .weights(weights),
        .biases(biases),
        .output_data(output_data)
    );

    initial begin
        $readmemh("conv2_input.mem", input_data);
        $readmemh("conv2_weights.mem", weights);
        $readmemh("conv2_bias_int32.mem", biases);

        #10;

        outfile = $fopen("conv2_output.mem", "w");

        for (i = 0; i < 3136; i = i + 1)
            $fdisplay(outfile, "%0d", output_data[i]);

        $fclose(outfile);

        $display("Conv2 outputs generated: 3136");
        $display("First output: %0d", output_data[0]);
        $display("Saved to conv2_output.mem");

        $finish;
    end

endmodule
