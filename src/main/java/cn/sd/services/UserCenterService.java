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
import java.util.UUID;


@Service
public class UserCenterService {

    private final UserRepository userRepository;
    private final FaceService faceService;

    @Autowired
    public UserCenterService(UserRepository userRepository, FaceService faceService) {
        this.userRepository = userRepository;
        this.faceService = faceService;
    }

    public HashMap<String, Object> findUserByEmailAndPasswordOrThrow(String email, String password) throws BusinessException, UnsupportedEncodingException {
        User user = userRepository.findByEmailAndPassword(email, password)
                .orElseThrow(() -> new BusinessException("账户或密码错误"));
        if (user != null) {
            return new HashMap<String, Object>() {{
                put("token", JwtUtil.build(user));
                put("user", user.hidePassword());
            }};
        } else {
            return null;
        }
    }

    public User isRegisterUser(User info) {
        User user = new User();
        user.setEmail(info.getEmail());
        user.setPassword(info.getPassword());
        user.setName(info.getName());
        String faceSetId = faceService.createFaceSet(UUID.randomUUID().toString().replaceAll("-", "").substring(0, 10));
        user.setAirSetId(String.valueOf(Double.valueOf(faceSetId).intValue()));
        return userRepository.save(user);
    }

    public Boolean isUserExits(String email) {
        List<User> user = userRepository.findByEmail(email);
        return user.size() != 0;
    }

    public String findAirSetIdByUserId(Long userId) throws BusinessException {
        User user = userRepository.findById(userId).orElseThrow(() -> new BusinessException("用户不存在"));
        return user.getAirSetId();
    }

}
