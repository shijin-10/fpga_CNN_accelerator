`timescale 1ns/1ps

module relu_tb;

    reg signed [31:0] data_in;
    wire signed [31:0] data_out;


    relu uut (
        .data_in(data_in),
        .data_out(data_out)
    );


    initial begin

        data_in = -3880;
        #10;

        $display(
            "Input = %d, ReLU Output = %d",
            data_in,
            data_out
        );


        data_in = 0;
        #10;

        $display(
            "Input = %d, ReLU Output = %d",
            data_in,
            data_out
        );


        data_in = 500;
        #10;

        $display(
            "Input = %d, ReLU Output = %d",
            data_in,
            data_out
        );


        data_in = -100;
        #10;

        $display(
            "Input = %d, ReLU Output = %d",
            data_in,
            data_out
        );


        $finish;

    end

endmodule
