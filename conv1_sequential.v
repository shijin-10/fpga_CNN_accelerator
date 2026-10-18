module conv1_sequential (
    input wire clk,
    input wire rst,
    input wire start,
    input wire signed [7:0] pixel_data,
    input wire signed [7:0] weight_data,
    input wire signed [31:0] bias,
    output reg [9:0] pixel_address,
    output reg [3:0] weight_address,
    output reg busy,
    output reg done,
    output reg output_valid,
    output reg [9:0] output_index,
    output reg signed [31:0] output_value
);

    localparam IDLE = 2'd0;
    localparam MAC  = 2'd1;
    localparam SAVE = 2'd2;

    reg [1:0] state;
    reg [9:0] pixel_index;
    reg [3:0] tap_index;
    reg signed [31:0] accumulator;
    reg signed [15:0] product;

    integer row, col, kernel_row, kernel_col;
    integer source_row, source_col, source_address;

    always @* begin
        row = pixel_index / 28;
        col = pixel_index % 28;
        kernel_row = tap_index / 3;
        kernel_col = tap_index % 3;

        source_row = row + kernel_row - 1;
        source_col = col + kernel_col - 1;

        source_address = 0;
        pixel_address = 0;
        weight_address = tap_index;
        product = 0;

        if (source_row >= 0 && source_row < 28 &&
            source_col >= 0 && source_col < 28) begin
            source_address = source_row * 28 + source_col;
            pixel_address = source_address;
            product = $signed(pixel_data) * $signed(weight_data);
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            busy <= 0;
            done <= 0;
            output_valid <= 0;
            pixel_index <= 0;
            tap_index <= 0;
            accumulator <= 0;
            output_index <= 0;
            output_value <= 0;
        end else begin
            done <= 0;
            output_valid <= 0;

            case (state)
                IDLE: begin
                    busy <= 0;
                    if (start) begin
                        busy <= 1;
                        pixel_index <= 0;
                        tap_index <= 0;
                        accumulator <= 0;
                        state <= MAC;
                    end
                end

                MAC: begin
                    busy <= 1;
                    if (tap_index < 8) begin
                        accumulator <= accumulator +
                                       {{16{product[15]}}, product};
                        tap_index <= tap_index + 1'b1;
                    end else begin
                        output_value <= accumulator +
                                        {{16{product[15]}}, product} +
                                        bias;
                        output_index <= pixel_index;
                        output_valid <= 1;
                        state <= SAVE;
                    end
                end

                SAVE: begin
                    if (pixel_index == 783) begin
                        busy <= 0;
                        done <= 1;
                        state <= IDLE;
                    end else begin
                        pixel_index <= pixel_index + 1'b1;
                        tap_index <= 0;
                        accumulator <= 0;
                        state <= MAC;
                    end
                end

                default: begin
                    state <= IDLE;
                    busy <= 0;
                end
            endcase
        end
    end

endmodule
