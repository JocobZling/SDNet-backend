package cn.sd.entities.bo;

import lombok.*;

import java.util.List;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder

public class UserUploadPhotoBo {

    private long userId;
    private List<String> photoList;

}
