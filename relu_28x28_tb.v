module relu_28x28_tb;

    reg signed [31:0] data_in [0:783];
    wire signed [31:0] data_out [0:783];

    integer i;

    relu_28x28 dut (
        .data_in(data_in),
        .data_out(data_out)
    );

    initial begin

        // Test values
        data_in[0] = 1261;
        data_in[1] = 977;
        data_in[2] = -410;
        data_in[3] = -650;
        data_in[4] = 920;

        // Fill remaining inputs with zero
        for (i = 5; i < 784; i = i + 1)
            data_in[i] = 0;

        #10;

        $display("ReLU results:");

        for (i = 0; i < 5; i = i + 1)
            $display("Input = %0d  ->  Output = %0d",
                     data_in[i], data_out[i]);

        $finish;

    end

endmodule

