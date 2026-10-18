module conv2_sequential (
    input wire clk,
    input wire rst,
    input wire start,

    input wire signed [7:0] input_data,
    input wire signed [7:0] weight_data,
    input wire signed [31:0] bias_data,

    output reg [10:0] input_address,
    output reg [10:0] weight_address,
    output reg [3:0] bias_address,

    output reg [11:0] output_address,
    output reg signed [31:0] output_value,

    output reg busy,
    output reg done,
    output reg output_valid
);

    localparam IDLE = 2'd0;
    localparam MAC  = 2'd1;
    localparam SAVE = 2'd2;

    reg [1:0] state;

    reg [3:0] filter_index;
    reg [7:0] pixel_index;
    reg [6:0] tap_index;

    reg signed [31:0] accumulator;
    reg signed [15:0] product;

    integer y, x, c, ky, kx;
    integer source_y, source_x;
    integer input_idx, weight_idx;

    always @* begin
        y = pixel_index / 14;
        x = pixel_index % 14;

        c  = tap_index / 9;
        ky = (tap_index % 9) / 3;
        kx = tap_index % 3;

        source_y = y + ky - 1;
        source_x = x + kx - 1;

        input_idx  = c * 196 + source_y * 14 + source_x;
        weight_idx = filter_index * 72 + tap_index;

        input_address  = 0;
        weight_address = weight_idx[10:0];
        bias_address   = filter_index;
        product        = 0;

        if (source_y >= 0 && source_y < 14 &&
            source_x >= 0 && source_x < 14) begin
            input_address = input_idx[10:0];

            product = $signed(input_data) *
                      $signed(weight_data);
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state         <= IDLE;
            filter_index  <= 0;
            pixel_index   <= 0;
            tap_index     <= 0;
            accumulator   <= 0;
            output_address <= 0;
            output_value  <= 0;
            busy          <= 0;
            done          <= 0;
            output_valid  <= 0;
        end else begin
            done         <= 0;
            output_valid <= 0;

            case (state)

                IDLE: begin
                    busy <= 0;

                    if (start) begin
                        filter_index <= 0;
                        pixel_index  <= 0;
                        tap_index    <= 0;
                        accumulator  <= 0;
                        busy         <= 1;
                        state        <= MAC;
                    end
                end

                MAC: begin
                    busy <= 1;

                    if (tap_index == 71) begin
                        if ($signed(accumulator) +
                            $signed(product) +
                            $signed(bias_data) < 0)
                            output_value <= 0;
                        else
                            output_value <= accumulator +
                                            {{16{product[15]}}, product} +
                                            bias_data;

                        output_address <= filter_index * 196 +
                                          pixel_index;
                        output_valid <= 1;
                        state <= SAVE;
                    end else begin
                        accumulator <= accumulator +
                                       {{16{product[15]}}, product};
                        tap_index <= tap_index + 1'b1;
                    end
                end

                SAVE: begin
                    if (pixel_index == 195) begin
                        pixel_index <= 0;

                        if (filter_index == 15) begin
                            busy  <= 0;
                            done  <= 1;
                            state <= IDLE;
                        end else begin
                            filter_index <= filter_index + 1'b1;
                            tap_index <= 0;
                            accumulator <= 0;
                            state <= MAC;
                        end
                    end else begin
                        pixel_index <= pixel_index + 1'b1;
                        tap_index <= 0;
                        accumulator <= 0;
                        state <= MAC;
                    end
                end

                default: begin
                    state <= IDLE;
                    busy  <= 0;
                end

            endcase
        end
    end

endmodule
