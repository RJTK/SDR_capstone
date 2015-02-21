Library UNISIM;
use UNISIM.vcomponents.all;

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity PWM_Audio is
  Generic ( AUDIO_WIDTH : INTEGER := 12; -- Audio bit width
            N : INTEGER := 8; -- Counter bits, f_pwm = f_clk / 2**N
            CLK_DIV : INTEGER := 29; -- Clock Division
            CLK_MUL : INTEGER := 128; -- Clock multiplication
            AUDIO_DIV : INTEGER := 2560; -- Audio decimation rate
            AUDIO_DIV_BITS : INTEGER := 12); -- ceiling of log2(AUDIO_DIV)

  Port ( CLK_IN: in STD_LOGIC; -- Input clock, probably 29MHz
         RST: in STD_LOGIC; -- Reset signal
         AUDIO_IN: in STD_LOGIC_VECTOR(AUDIO_WIDTH - 1 downto 0); -- Audio input sample
         PWM_OUT: out STD_LOGIC
         );
end PWM_Audio;

architecture Behavioral of PWM_Audio is

signal CLK : STD_LOGIC := '0';

signal audio_sample : STD_LOGIC_VECTOR(N - 1 downto 0) := (others => '0');
signal audio_count : STD_LOGIC_VECTOR(AUDIO_DIV_BITS - 1 downto 0) := (others => '0');
signal load_audio : STD_LOGIC := '0';

signal PWM_count : STD_LOGIC_VECTOR(N - 1 downto 0) := (others => '0');
signal PWM_set : STD_LOGIC := '0';
signal PWM_rst : STD_LOGIC := '1';

begin

DCM_CLKGEN_inst : DCM_CLKGEN
  generic map (
    CLKFXDV_DIVIDE => 2,       -- CLKFXDV divide value (2, 4, 8, 16, 32)
    CLKFX_DIVIDE => CLK_DIV,         -- Divide value - D - (1-256)
    CLKFX_MD_MAX => 0.0,       -- Specify maximum M/D ratio for timing anlysis
    CLKFX_MULTIPLY => CLK_MUL,       -- Multiply value - M - (2-256)
    CLKIN_PERIOD => 34.4,       -- Input clock period specified in nS
    SPREAD_SPECTRUM => "NONE", -- Spread Spectrum mode "NONE", "CENTER_LOW_SPREAD", "CENTER_HIGH_SPREAD",
                                 -- "VIDEO_LINK_M0", "VIDEO_LINK_M1" or "VIDEO_LINK_M2" 
    STARTUP_WAIT => TRUE       -- Delay config DONE until DCM_CLKGEN LOCKED (TRUE/FALSE)
   )
  port map (
    CLKFX => CLK,         -- 1-bit output: Generated clock output
    CLKFX180 => open,   -- 1-bit output: Generated clock output 180 degree out of phase from CLKFX.
    CLKFXDV => open,     -- 1-bit output: Divided clock output
    LOCKED => open,       -- 1-bit output: Locked output
    PROGDONE => open,   -- 1-bit output: Active high output to indicate the successful re-programming
    STATUS => open,       -- 2-bit output: DCM_CLKGEN status
    CLKIN => CLK_IN,         -- 1-bit input: Input clock
    FREEZEDCM => open, -- 1-bit input: Prevents frequency adjustments to input clock
    PROGCLK => open,     -- 1-bit input: Clock input for M/D reconfiguration
    PROGDATA => open,   -- 1-bit input: Serial data input for M/D reconfiguration
    PROGEN => open,       -- 1-bit input: Active high program enable
    RST => open              -- 1-bit input: Reset input pin
    );

-- A counter to decimate the audio input
cnt_audio_counter: process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (RST = '0') then
      audio_count <= (others => '0');
    elsif (audio_count = AUDIO_DIV) then
      load_audio <= '1';
      audio_count <= (others => '0');
    else
      load_audio <= '0';
      audio_count <= audio_count + 1;
    end if;
  end if;
end process cnt_audio_counter;

-- Audio sample input register
-- This register is crossing clock domains
-- I don't know how much trouble this could cause, if any.
-- AUDIO_IN comes from 29MHz, 
reg_audio_sample: process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (RST = '0') then
      audio_sample <= (others => '0');
    elsif (load_audio = '1') then
      -- Truncate audio samples
      -- I could round here but this is easy
      audio_sample(N - 1 downto 0) <= AUDIO_IN(AUDIO_WIDTH - 1 downto AUDIO_WIDTH - N);
    end if;
  end if;
end process reg_audio_sample;

-- PWM counter, this defines f_pwm = f_clk / 2**N
cnt_pwm: process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (RST = '0') then
      PWM_count <= (others => '0');
    elsif (PWM_count = (PWM_count'range => '1')) then
      PWM_count <= (others => '0');
    else
      PWM_count <= PWM_count + 1;
    end if;
  end if;
end process cnt_pwm;

-- Compares the pwm count to the audio sample
-- Sets the output register if PWM_count is full
-- Resets the output register is PWM_count = audio_sample
cmp_pwm: process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (RST = '0') then
      PWM_set <= '0';
      PWM_rst <= '1';
    elsif (PWM_count = audio_sample) then
      PWM_set <= '0';
      PWM_rst <= '1';
    elsif (PWM_count = (PWM_count'range => '0')) then
      PWM_set <= '1';
      PWM_rst <= '0';
    end if;
  end if;
end process cmp_pwm;

-- PWM output register
reg_pwm_out: process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (PWM_set = '1') then
      PWM_OUT <= '1';
    elsif (PWM_rst = '1') then
      PWM_OUT <= '0';
    end if;
  end if;
end process reg_pwm_out;

end Behavioral;

