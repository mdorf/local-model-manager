import plistlib

from lmm.deploy import (
    LABEL,
    account_steps,
    acl_steps,
    find_free_service_uid,
    install_steps,
    launchd_plist,
    plist_install_path,
    uninstall_steps,
)


def test_plist_has_required_keys():
    xml = launchd_plist(exec_path="/usr/local/bin/lmm", host="127.0.0.1",
                        port=8770, user="_lmm")
    data = plistlib.loads(xml.encode())
    assert data["Label"] == LABEL
    assert data["UserName"] == "_lmm"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["ProgramArguments"][0] == "/usr/local/bin/lmm"
    assert "daemon" in data["ProgramArguments"]
    assert "--host" in data["ProgramArguments"]
    assert "127.0.0.1" in data["ProgramArguments"]


def test_account_steps_create_hidden_no_shell_account():
    joined = "\n".join(account_steps(user="_lmm", uid=251))
    assert "dscl . -create /Users/_lmm" in joined
    assert "/usr/bin/false" in joined
    assert "IsHidden" in joined
    assert "251" in joined
    assert "PrimaryGroupID" in joined


def test_acl_steps_grant_read_only():
    joined = "\n".join(acl_steps(user="_lmm", models_dir="/Users/Shared/models"))
    assert "chmod" in joined and "+a" in joined
    assert "_lmm allow" in joined
    assert "read" in joined
    assert "write" not in joined
    assert "/Users/Shared/models" in joined


def test_install_steps_compose_all_phases():
    joined = "\n".join(install_steps(exec_path="/usr/local/bin/lmm", user="_lmm",
                                     uid=251, host="127.0.0.1", port=8770,
                                     models_dir="/Users/Shared/models"))
    assert "dscl . -create /Users/_lmm" in joined
    assert "+a" in joined
    assert plist_install_path() in joined
    assert "launchctl bootstrap" in joined
    assert "socketfilterfw" in joined


def test_uninstall_steps_remove():
    joined = "\n".join(uninstall_steps(user="_lmm"))
    assert "launchctl bootout" in joined
    assert plist_install_path() in joined
    assert "dscl . -delete /Users/_lmm" in joined


def test_find_free_service_uid_returns_unused_int():
    uid = find_free_service_uid()
    assert isinstance(uid, int)
    assert 200 <= uid < 500
