
`timescale 1ns/1ps

module requantize_conv1_tb;

    reg signed [31:0] input_data [0:1567];
    wire signed [7:0] output_data [0:1567];

    integer i;
    integer outfile;

    requantize_array #(
        .N(1568),
        .MULTIPLIER(64'd3063111),
        .SHIFT(30)
    ) dut (
        .input_data(input_data),
        .output_data(output_data)
    );

    initial begin
        $readmemh("conv1_pool_output_hex.mem", input_data);

        #10;
        outfile = $fopen("conv2_input_requantized.mem", "w");

        for (i = 0; i < 1568; i = i + 1)
            $fdisplay(outfile, "%0d", output_data[i]);

        $fclose(outfile);
        $display("Conv1 requantization completed: 1568 values");
        $finish;
    end

endmodule
