package cn.sd.services;

import cn.sd.entities.User;
import cn.sd.exceptions.BusinessException;
import cn.sd.repositories.UserRepository;
import cn.sd.utils.JwtUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.List;


@Service
public class UserCenterService {

    private final UserRepository userRepository;

    @Autowired
    public UserCenterService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    // BusinessException业务异常类
    public HashMap<String, Object> findUserByEmailAndPasswordOrThrow(String email, String password) throws BusinessException, UnsupportedEncodingException {
        User user = userRepository.findByEmailAndPassword(email, password)
                .orElseThrow(() -> new BusinessException("账户或密码错误"));
        if (user != null) {
            return new HashMap<String, Object>() {{
                put("token", JwtUtil.build(user));
                put("user", user.hidePassword()); // 把没有密码的user对象存起来
            }};
        } else {
            return null;
        }
    }

    public HashMap<String, Object> isRegisterUser(User info) throws UnsupportedEncodingException {
        User user = new User();
        user.setEmail(info.getEmail());
        user.setPassword(info.getPassword());
        user.setName(info.getName());
        userRepository.save(user);
        return new HashMap<String, Object>(){{
            put("token", JwtUtil.build(user));
            put("user", user.hidePassword());
        }};
    }

    // 只要邮箱不冲突就可以注册
    public Boolean isUserExits(String email) {
        List<User> user = userRepository.findByEmail(email);
        return user.size() != 0;
    }

    public User authUserOldPasswordUpdatePassword(Long userId,String oldPassword,String newPassword)throws BusinessException{
        User foundUser=userRepository.findByIdAndPassword(userId,oldPassword).orElseThrow(
                ()->new BusinessException("旧密码错误")
        );
        foundUser.updatePassword(newPassword);
        return userRepository.save(foundUser);
    }

    public HashMap<String,Object> updateUserById(Long userId, User user)throws BusinessException,UnsupportedEncodingException {
        User foundUser=userRepository.findById(userId).orElseThrow(
                ()->new BusinessException("Current user is not exist")
        );
        foundUser.update(user);
        User userNew=userRepository.save(foundUser);
        userNew.hidePassword();
        return new HashMap<String, Object>(){{
            put("token",JwtUtil.build(userNew));
            put("user",userNew);
        }};
    }
}
