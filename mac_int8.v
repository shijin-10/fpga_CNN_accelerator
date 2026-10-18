module mac_int8 (
    input  signed [7:0]  a,
    input  signed [7:0]  b,
    input signed [31:0] acc_in,
    output signed [31:0] acc_out
);

    wire signed [15:0] product;

    assign product = a * b;

    assign acc_out = acc_in + product;

endmodule
