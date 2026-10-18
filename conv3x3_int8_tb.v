`timescale 1ns/1ps

module conv3x3_int8_tb;

    reg signed [7:0] p0, p1, p2;
    reg signed [7:0] p3, p4, p5;
    reg signed [7:0] p6, p7, p8;

    reg signed [7:0] w0, w1, w2;
    reg signed [7:0] w3, w4, w5;
    reg signed [7:0] w6, w7, w8;

    reg signed [31:0] bias;

    wire signed [31:0] result;


    conv3x3_int8 uut (

        .p0(p0),
        .p1(p1),
        .p2(p2),
        .p3(p3),
        .p4(p4),
        .p5(p5),
        .p6(p6),
        .p7(p7),
        .p8(p8),

        .w0(w0),
        .w1(w1),
        .w2(w2),
        .w3(w3),
        .w4(w4),
        .w5(w5),
        .w6(w6),
        .w7(w7),
        .w8(w8),

        .bias(bias),

        .result(result)
    );


    initial begin

        // Image window
        p0 = 1;
        p1 = 2;
        p2 = 3;

        p3 = 4;
        p4 = 5;
        p5 = 6;

        p6 = 7;
        p7 = 8;
        p8 = 9;


        // Kernel
        w0 = 1;
        w1 = 1;
        w2 = 1;

        w3 = 1;
        w4 = 1;
        w5 = 1;

        w6 = 1;
        w7 = 1;
        w8 = 1;


        // Bias
        bias = 10;

        #10;

        $display(
            "Convolution result = %d",
            result
        );

        $finish;

    end

endmodule
	
