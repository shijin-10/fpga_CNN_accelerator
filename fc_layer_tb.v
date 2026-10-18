
module fc_layer_tb;

    reg signed [31:0] input_data [0:783];
    reg signed [7:0]  weights [0:7839];
    reg signed [31:0] biases [0:9];

    wire signed [63:0] output_data [0:9];

    integer i;
    integer outfile;

    fc_layer dut (
        .input_data(input_data),
        .weights(weights),
        .biases(biases),
        .output_data(output_data)
    );

    initial begin
        $readmemh("fc_input.mem", input_data);
        $readmemh("fc_weights.mem", weights);
        $readmemh("fc_bias_int32.mem", biases);

        #10;

        outfile = $fopen("fc_output.mem", "w");

        for (i = 0; i < 10; i = i + 1)
            $fdisplay(outfile, "%0d", output_data[i]);

        $fclose(outfile);

        $display("FC outputs generated: 10");

        for (i = 0; i < 10; i = i + 1)
            $display("Digit %0d score: %0d", i, output_data[i]);

        $finish;
    end

endmodule
