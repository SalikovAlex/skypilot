"""Constants used by the GCP provisioner."""

VERSION = 'v1'
# Using v2 according to
# https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-curl # pylint: disable=line-too-long
TPU_VM_VERSION = 'v2'

RAY = 'ray-autoscaler'
DEFAULT_SERVICE_ACCOUNT_ID = RAY + '-sa-' + VERSION
SERVICE_ACCOUNT_EMAIL_TEMPLATE = (
    '{account_id}@{project_id}.iam.gserviceaccount.com')
DEFAULT_SERVICE_ACCOUNT_CONFIG = {
    'displayName': f'Ray Autoscaler Service Account ({VERSION})',
}

SKYPILOT = 'skypilot'
SKYPILOT_SERVICE_ACCOUNT_ID = SKYPILOT + '-' + VERSION
SKYPILOT_SERVICE_ACCOUNT_EMAIL_TEMPLATE = (
    '{account_id}@{project_id}.iam.gserviceaccount.com')
SKYPILOT_SERVICE_ACCOUNT_CONFIG = {
    'displayName': f'SkyPilot Service Account ({VERSION})',
}

# Those roles will be always added.
# NOTE: `serviceAccountUser` allows the head node to create workers with
# a serviceAccount. `roleViewer` allows the head node to run bootstrap_gcp.
DEFAULT_SERVICE_ACCOUNT_ROLES = [
    'roles/storage.admin',
    'roles/compute.admin',
    'roles/iam.serviceAccountUser',
    'roles/iam.roleViewer',
]
# Those roles will only be added if there are TPU nodes defined in config.
TPU_SERVICE_ACCOUNT_ROLES = ['roles/tpu.admin']

# If there are TPU nodes in config, this field will be set
# to True in config['provider'].
HAS_TPU_PROVIDER_FIELD = '_has_tpus'

# NOTE: iam.serviceAccountUser allows the Head Node to create worker nodes
# with ServiceAccounts.

SKYPILOT_VPC_NAME = 'skypilot-vpc'
SKYPILOT_GPU_DIRECT_VPC_NUM = 5
SKYPILOT_GPU_DIRECT_VPC_CIDR_PREFIX = '10.129'
GPU_DIRECT_TCPX_INSTANCE_TYPES = [
    'a3-edgegpu-8g',
    'a3-highgpu-8g',
]
# The prefix length of the cluster name.
# To make sure the VPC and subnet names are within the GCP limits.
CLUSTER_PREFIX_LENGTH = 10

COMPACT_GROUP_PLACEMENT_POLICY = 'compact'
COLLOCATED_COLLOCATION = 'COLLOCATED'
GPU_DIRECT_TCPX_USER_DATA = """#!/bin/bash
    set -e
    set -x
    # Install GPU Direct TCPX
    cos-extensions install gpu -- --version=latest;
    sudo mount --bind /var/lib/nvidia /var/lib/nvidia;
    sudo mount -o remount,exec /var/lib/nvidia;
    docker ps -a | grep -q receive-datapath-manager || \
    docker run \
    --detach \
    --pull=always \
    --name receive-datapath-manager \
    --privileged \
    --cap-add=NET_ADMIN --network=host \
    --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
    --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 \
    --device /dev/nvidia2:/dev/nvidia2 --device /dev/nvidia3:/dev/nvidia3 \
    --device /dev/nvidia4:/dev/nvidia4 --device /dev/nvidia5:/dev/nvidia5 \
    --device /dev/nvidia6:/dev/nvidia6 --device /dev/nvidia7:/dev/nvidia7 \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidiactl:/dev/nvidiactl \
    --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    --volume /run/tcpx:/run/tcpx \
    --entrypoint /tcpgpudmarxd/build/app/tcpgpudmarxd \
    us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/tcpgpudmarxd \
    --gpu_nic_preset a3vm --gpu_shmem_type fd --uds_path "/run/tcpx" --setup_param "--verbose 128 2 0";
    sudo iptables -I INPUT -p tcp -m tcp -j ACCEPT;
    docker run --rm -v /var/lib:/var/lib us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx install --install-nccl;
    sudo mount --bind /var/lib/tcpx /var/lib/tcpx;
    sudo mount -o remount,exec /var/lib/tcpx;
    echo "GPU Direct TCPX installed"
    """

GPU_DIRECT_TCPX_SPECIFIC_OPTIONS = [
    '--cap-add=IPC_LOCK',
    '--userns=host',
    '--volume /run/tcpx:/run/tcpx',
    '--volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64',
    '--volume /var/lib/tcpx/lib64:/usr/local/tcpx/lib64',
    '--volume /var/lib/nvidia/bin:/usr/local/nvidia/bin',
    '--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864',
    '--device /dev/nvidia0:/dev/nvidia0',
    '--device /dev/nvidia1:/dev/nvidia1',
    '--device /dev/nvidia2:/dev/nvidia2',
    '--device /dev/nvidia3:/dev/nvidia3',
    '--device /dev/nvidia4:/dev/nvidia4',
    '--device /dev/nvidia5:/dev/nvidia5',
    '--device /dev/nvidia6:/dev/nvidia6',
    '--device /dev/nvidia7:/dev/nvidia7',
    '--device /dev/nvidia-uvm:/dev/nvidia-uvm',
    '--device /dev/nvidiactl:/dev/nvidiactl',
    '--env LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/tcpx/lib64',
]

# Below parameters are from the default VPC on GCP.
# https://cloud.google.com/vpc/docs/firewalls#more_rules_default_vpc
VPC_TEMPLATE: dict = {
    'name': '{VPC_NAME}',
    'selfLink': 'projects/{PROJ_ID}/global/networks/{VPC_NAME}',
    'autoCreateSubnetworks': True,
    'mtu': 1460,
    'routingConfig': {
        'routingMode': 'GLOBAL'
    },
}
# Required firewall rules for SkyPilot to work.
FIREWALL_RULES_REQUIRED = [
    # Allow internal connections between GCP VMs for Ray multi-node cluster.
    {
        'direction': 'INGRESS',
        'allowed': [
            {
                'IPProtocol': 'tcp',
                'ports': ['0-65535']
            },
            {
                'IPProtocol': 'udp',
                'ports': ['0-65535']
            },
        ],
        'sourceRanges': ['10.128.0.0/9'],
    },
    # Allow ssh connection from anywhere.
    {
        'direction': 'INGRESS',
        'allowed': [{
            'IPProtocol': 'tcp',
            'ports': ['22'],
        }],
        # TODO(skypilot): some users reported that this should be relaxed (e.g.,
        # allowlisting only certain IPs to have ssh access).
        'sourceRanges': ['0.0.0.0/0'],
    },
]

# Template when creating firewall rules for a new VPC.
FIREWALL_RULES_TEMPLATE = [
    {
        'name': '{VPC_NAME}-allow-custom',
        'description': ('Allows connection from any source to any instance on '
                        'the network using custom protocols.'),
        'network': 'projects/{PROJ_ID}/global/networks/{VPC_NAME}',
        'selfLink':
            ('projects/{PROJ_ID}/global/firewalls/{VPC_NAME}-allow-custom'),
        'direction': 'INGRESS',
        'priority': 65534,
        'allowed': [
            {
                'IPProtocol': 'tcp',
                'ports': ['0-65535']
            },
            {
                'IPProtocol': 'udp',
                'ports': ['0-65535']
            },
            {
                'IPProtocol': 'icmp'
            },
        ],
        'sourceRanges': ['10.128.0.0/9'],
    },
    {
        'name': '{VPC_NAME}-allow-ssh',
        'description':
            ('Allows TCP connections from any source to any instance on the '
             'network using port 22.'),
        'network': 'projects/{PROJ_ID}/global/networks/{VPC_NAME}',
        'selfLink': 'projects/{PROJ_ID}/global/firewalls/{VPC_NAME}-allow-ssh',
        'direction': 'INGRESS',
        'priority': 65534,
        'allowed': [{
            'IPProtocol': 'tcp',
            'ports': ['22'],
        }],
        # TODO(skypilot): some users reported that this should be relaxed (e.g.,
        # allowlisting only certain IPs to have ssh access).
        'sourceRanges': ['0.0.0.0/0'],
    },
    {
        'name': '{VPC_NAME}-allow-icmp',
        'description': ('Allows ICMP connections from any source to any '
                        'instance on the network.'),
        'network': 'projects/{PROJ_ID}/global/networks/{VPC_NAME}',
        'selfLink': 'projects/{PROJ_ID}/global/firewalls/{VPC_NAME}-allow-icmp',
        'direction': 'INGRESS',
        'priority': 65534,
        'allowed': [{
            'IPProtocol': 'icmp',
        }],
        'sourceRanges': ['0.0.0.0/0'],
    },
]

GCP_MINIMAL_PERMISSIONS = [
    'serviceusage.services.enable',
    'serviceusage.services.list',
]

# A list of permissions required to run SkyPilot on GCP.
# Keep this in sync with https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/gcp.html # pylint: disable=line-too-long
VM_MINIMAL_PERMISSIONS = [
    'compute.disks.create',
    'compute.disks.list',
    'compute.firewalls.get',
    'compute.instances.create',
    'compute.instances.delete',
    'compute.instances.get',
    'compute.instances.list',
    'compute.instances.setLabels',
    'compute.instances.setServiceAccount',
    'compute.instances.start',
    'compute.instances.stop',
    'compute.networks.get',
    'compute.networks.list',
    'compute.networks.getEffectiveFirewalls',
    'compute.globalOperations.get',
    'compute.subnetworks.use',
    'compute.subnetworks.list',
    'compute.subnetworks.useExternalIp',
    'compute.projects.get',
    'compute.zoneOperations.get',
    'iam.roles.get',
    # We now skip the check for `iam.serviceAccounts.actAs` permission for
    # simplicity as it can be granted at the service-account level.
    # Check: sky.provision.gcp.config::_is_permission_satisfied
    # 'iam.serviceAccounts.actAs',
    'iam.serviceAccounts.get',
    'serviceusage.services.use',
    'resourcemanager.projects.get',
    'resourcemanager.projects.getIamPolicy',
]

STORAGE_MINIMAL_PERMISSIONS = [
    'storage.buckets.create',
    'storage.buckets.delete',
]

# Permissions implied by GCP built-in roles. We hardcode these here, as we
# cannot get the permissions of built-in role from the GCP Python API.
# The lists are not exhaustive, but should cover the permissions listed in
# VM_MINIMAL_PERMISSIONS.
# Check: sky.provision.gcp.config::_is_permission_satisfied
BUILTIN_ROLE_TO_PERMISSIONS = {
    'roles/iam.serviceAccountUser': ['iam.serviceAccounts.actAs'],
    'roles/iam.serviceAccountViewer': [
        'iam.serviceAccounts.get', 'iam.serviceAccounts.getIamPolicy'
    ],
    # TODO(zhwu): Add more built-in roles to make the permission check more
    # robust.
}

FIREWALL_PERMISSIONS = [
    'compute.firewalls.create',
    'compute.firewalls.delete',
]

RESERVATION_PERMISSIONS = [
    'compute.reservations.list',
]

TPU_MINIMAL_PERMISSIONS = [
    'tpu.nodes.create',
    'tpu.nodes.delete',
    'tpu.nodes.list',
    'tpu.nodes.get',
    'tpu.nodes.update',
    'tpu.operations.get',
]

# The maximum number of times to poll for the status of an operation.
POLL_INTERVAL = 1
MAX_POLLS = 60 // POLL_INTERVAL
# Stopping instances can take several minutes, so we increase the timeout
MAX_POLLS_STOP = MAX_POLLS * 8

# MIG constants
MANAGED_INSTANCE_GROUP_CONFIG = 'managed-instance-group'
DEFAULT_MANAGED_INSTANCE_GROUP_PROVISION_TIMEOUT = 900  # 15 minutes
MIG_NAME_PREFIX = 'sky-mig-'
INSTANCE_TEMPLATE_NAME_PREFIX = 'sky-it-'
