package cn.sd.entities.vo;

import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserEmailAndPassword {
    String email;
    String password;
}
