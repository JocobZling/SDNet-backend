package cn.sd.entities.vo;

import cn.sd.entities.Photo;
import lombok.*;

import java.util.List;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder

public class PhotoDateAndSorce {
    String date;
    List<Photo> photoList;
}
