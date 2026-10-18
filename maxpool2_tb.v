
module maxpool2_tb;

    reg signed [31:0] input_data [0:3135];
    wire signed [31:0] output_data [0:783];

    integer i;
    integer outfile;

    maxpool_2x2_conv2 dut (
        .input_data(input_data),
        .output_data(output_data)
    );

    initial begin
        $readmemh("conv2_output_hex.mem", input_data);

        #10;

        outfile = $fopen("conv2_pool_output.mem", "w");

        for (i = 0; i < 784; i = i + 1)
            $fdisplay(outfile, "%0d", output_data[i]);

        $fclose(outfile);

        $display("MaxPool2 outputs generated: 784");
        $display("First output: %0d", output_data[0]);
        $finish;
    end

endmodule
