module maxpool_2x2_tb;

    reg signed [31:0] input_data [0:6271];
    wire signed [31:0] output_data [0:1567];

    integer i;
    integer row;
    integer col;

    maxpool_2x2 dut (
        .input_data(input_data),
        .output_data(output_data)
    );

    initial begin

        // Clear all inputs
        for (i = 0; i < 6272; i = i + 1)
            input_data[i] = 0;

        // ---------------------------------------------
        // Test Filter 0 with simple values
        // ---------------------------------------------

        // 28x28 Filter 0
        input_data[0] = 10;
        input_data[1] = 20;
        input_data[28] = 30;
        input_data[29] = 40;

        // Second 2x2 block
        input_data[2] = 50;
        input_data[3] = 60;
        input_data[30] = 70;
        input_data[31] = 80;

        #10;

        $display("");
        $display("MaxPool Filter 0:");
        $display("");

        $display("Output[0] = %0d", output_data[0]);
        $display("Output[1] = %0d", output_data[1]);

        $finish;

    end

endmodule
