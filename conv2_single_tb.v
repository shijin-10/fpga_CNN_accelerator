module conv2_single_tb;

    reg signed [7:0] input_data [0:71];
    reg signed [7:0] weights [0:71];

    reg signed [31:0] bias;

    wire signed [31:0] output_data;

    integer i;

    conv2_single dut (
        .input_data(input_data),
        .weights(weights),
        .bias(bias),
        .output_data(output_data)
    );

    initial begin

        // Test 72 input values
        for (i = 0; i < 72; i = i + 1)
            input_data[i] = i + 1;

        // All weights = 1
        for (i = 0; i < 72; i = i + 1)
            weights[i] = 1;

        // Bias
        bias = 10;

        #10;

        $display("Conv2 single output = %0d", output_data);

        $finish;

    end

endmodule

