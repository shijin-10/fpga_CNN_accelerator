`timescale 1ns/1ps

module requantize_conv2_tb;

    reg signed [31:0] input_data [0:783];
    wire signed [7:0] output_data [0:783];

    integer i;
    integer outfile;

    requantize_array #(
        .N(784),
        .MULTIPLIER(64'd2850408),
        .SHIFT(30)
    ) dut (
        .input_data(input_data),
        .output_data(output_data)
    );

    initial begin
        $readmemh("conv2_pool_output_hex.mem", input_data);

        #10;
        outfile = $fopen("fc_input_requantized.mem", "w");

        for (i = 0; i < 784; i = i + 1)
            $fdisplay(outfile, "%0d", output_data[i]);

        $fclose(outfile);
        $display("Conv2 requantization completed: 784 values");
        $finish;
    end

endmodule
