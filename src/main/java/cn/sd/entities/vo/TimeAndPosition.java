package cn.sd.entities.vo;

import lombok.*;

import java.util.HashMap;
import java.util.List;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TimeAndPosition {
    List<HashMap<String,String>> TAndP;
    String Position;

}
