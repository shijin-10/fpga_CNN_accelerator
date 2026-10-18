
module requantize_array #(
    parameter N = 1568,
    parameter signed [63:0] MULTIPLIER = 64'd3063111,
    parameter SHIFT = 30
)(
    input wire signed [31:0] input_data [0:N-1],
    output reg signed [7:0] output_data [0:N-1]
);

    integer i;
    reg signed [63:0] temp;
    reg signed [63:0] scaled;

    always @(*) begin
        for (i = 0; i < N; i = i + 1) begin
            temp = $signed(input_data[i]) * MULTIPLIER;
            temp = temp + (64'd1 << (SHIFT - 1));
            scaled = temp >>> SHIFT;

            if (scaled > 127)
                output_data[i] = 8'sd127;
            else if (scaled < 0)
                output_data[i] = 8'sd0;
            else
                output_data[i] = scaled[7:0];
        end
    end

endmodule
