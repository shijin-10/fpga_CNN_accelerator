
module mac_unit (
    input wire clk,
    input wire rst,
    input wire clear,
    input wire enable,
    input wire signed [7:0] data_in,
    input wire signed [7:0] weight,
    output reg signed [31:0] accumulator
);

    wire signed [15:0] product;
    assign product = data_in * weight;

    always @(posedge clk) begin
        if (rst)
            accumulator <= 32'sd0;
        else if (clear)
            accumulator <= 32'sd0;
        else if (enable)
            accumulator <= accumulator + {{16{product[15]}}, product};
    end

endmodule
