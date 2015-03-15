library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity sp601_test is
  Port ( USER_CLK : in STD_LOGIC;
         PB : in STD_LOGIC_VECTOR(3 downto 0);
         FMC_LA02_P : in STD_LOGIC;
         FMC_LA02_N : in STD_LOGIC;
         FMC_LA04_P : in STD_LOGIC;
         FMC_LA07_P : in STD_LOGIC;
         FMC_LA11_P : in STD_LOGIC;
         FMC_LA19_P : in STD_LOGIC;
         FMC_LA21_P : in STD_LOGIC;
         FMC_LA24_P : in STD_LOGIC;
         FMC_LA28_P : in STD_LOGIC;
         FMC_LA28_N : in STD_LOGIC;
         FMC_LA30_P : in STD_LOGIC;
         FMC_LA30_N : in STD_LOGIC;
         
         LED : out STD_LOGIC_VECTOR(3 downto 0);
         GPIO_HDR4 : out STD_LOGIC;
         FMC_LA32_P : out STD_LOGIC
         );
end sp601_test;

architecture Behavioral of sp601_test is

signal RST : STD_LOGIC;
signal CLK : STD_LOGIC;
signal audio : STD_LOGIC_VECTOR(11 downto 0);

component ADC_ctrl
  port ( D11 : in STD_LOGIC;
         D10 : in STD_LOGIC;
         D09 : in STD_LOGIC;
         D08 : in STD_LOGIC;
         D07 : in STD_LOGIC;
         D06 : in STD_LOGIC;
         D05 : in STD_LOGIC;
         D04 : in STD_LOGIC;
         D03 : in STD_LOGIC;
         D02 : in STD_LOGIC;
         D01 : in STD_LOGIC;
         D00 : in STD_LOGIC;
         CLK_IN : in STD_LOGIC;
         RST : in STD_LOGIC;
         DATA_OUT : out STD_LOGIC_VECTOR(11 downto 0);
         CLK_OUT : out STD_LOGIC);
end component;

component PWM_Audio
  Generic ( AUDIO_WIDTH : INTEGER := 12;
            N : INTEGER := 8;
            CLK_DIV : INTEGER := 29;
            CLK_MUL : INTEGER := 128;
            AUDIO_DIV : INTEGER := 2560;
            AUDIO_DIV_BITS : INTEGER := 12);
  port ( CLK_IN : in STD_LOGIC;
         RST : in STD_LOGIC;
         AUDIO_IN : in STD_LOGIC_VECTOR(11 downto 0);
         PWM_OUT : out STD_LOGIC
         );
end component;

begin

ADC : ADC_ctrl
  port map( D11 => FMC_LA02_P,
            D10 => FMC_LA02_N,
            D09 => FMC_LA04_P,
            D08 => FMC_LA07_P,
            D07 => FMC_LA11_P,
            D06 => FMC_LA19_P,
            D05 => FMC_LA21_P,
            D04 => FMC_LA24_P,
            D03 => FMC_LA28_P,
            D02 => FMC_LA28_N,
            D01 => FMC_LA30_P,
            D00 => FMC_LA30_N,
            CLK_IN => USER_CLK,
            RST => RST,
            DATA_OUT => audio,
            CLK_OUT => FMC_LA32_P
            );

PWM : PWM_Audio
  generic map( AUDIO_WIDTH => 12,
               N => 8,
               CLK_DIV => 27,
               CLK_MUL => 128,
               AUDIO_DIV => 2560,
               AUDIO_DIV_BITS => 12
               )
  port map( CLK_IN => USER_CLK,
            RST => RST,
            AUDIO_IN => audio,
            PWM_OUT => GPIO_HDR4
            );

CLK_input : process(USER_CLK)
begin
  CLK <= USER_CLK;
end process CLK_input;

rst_test : process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (PB(0) = '1') then
      RST <= '0';
      LED(0) <= '1';
    else
      RST <= '1';
      LED(0) <= '0';
    end if;
  end if;
end process rst_test;

test : process(CLK)
begin
  if (CLK'Event and CLK = '1') then
    if (PB(1) = '1') then
      LED(1) <= '1';
    elsif (PB(2) = '1') then
      LED(2) <= '1';
    elsif (PB(3) = '1') then
      LED(3) <= '1';
    else
      LED(3 downto 1) <= "000";
    end if;
  end if;
end process test;

end Behavioral;

