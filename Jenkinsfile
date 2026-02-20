def VERSION
def APPLICATION_IMAGE
def ARTIFACT_RELEASED = false

pipeline {
    agent {
        label 'built-in'
    }
    parameters {
        booleanParam(
            name: "RELEASE",
            defaultValue: false,
            description: "Release a non-snapshot version (Only affects builds on main)"
        )
        choice(
            name: 'VERSION_INCREMENT',
            choices: ['Patch', 'Minor', 'Major'],
            description: 'Which version part to increment (Only affects builds on main when releasing)'
        )
    }
    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    environment {
        SSH_CREDENTIALS                 = '022fef2f-afaa-482c-884a-9d2281fbc5cc'
        APPLICATION_IMAGE               = 'slt-renewal-model'
        APPLICATION_REGISTRY            = '958146224438.dkr.ecr.us-west-2.amazonaws.com'
        APPLICATION_REGISTRY_CREDENTIAL = 'ecr:us-west-2:fts-ecr'

        APPLICATION_VERSION_FILE        = 'slt_season_renewal/_version.py'
    }

    stages {
        stage('Package Info') {
            agent {
                docker {
                    image '958146224438.dkr.ecr.us-west-2.amazonaws.com/docker-libraries:python3.10-dev'
                    label 'docker'
                }
            }
            steps {
                script {
                    if (env.BRANCH_NAME == 'main') {
                        bumpCommitMessage = sh(
                            returnStatus: true, 
                            script: "git log -n 1 | tr '\\n' ' ' | grep '.*FTSCloudOps.*bump'"
                        )
                        if ( bumpCommitMessage == 0 ) {
                            currentBuild.result = 'ABORTED'
                            throw new hudson.AbortException('last commit was a bump.  Aborting build.')
                        }
                        if (params.VERSION_INCREMENT == 'Patch') {
                            sh "bump2version --verbose patch ./${env.APPLICATION_VERSION_FILE}"
                        } else if (params.VERSION_INCREMENT == 'Minor') {
                            sh "bump2version --verbose minor ./${env.APPLICATION_VERSION_FILE}"
                        } else if (params.VERSION_INCREMENT == 'Major') {
                            sh "bump2version --verbose major ./${env.APPLICATION_VERSION_FILE}"
                        } else {
                            error "Unexpected value of VERSION_INCREMENT: " + params.VERSION_INCREMENT
                        }
                        VERSION = sh(
                            script: "cat ./${env.APPLICATION_VERSION_FILE} | grep version | cut -d \"\\\"\" -f2",
                            returnStdout: true
                        ).trim()
                    } else {
                        OLD_VERSION = sh(
                                script: "cat ./${env.APPLICATION_VERSION_FILE} | grep version | cut -d \"\\\"\" -f2",
                                returnStdout: true
                        ).trim()
                        VERSION     = env.CHANGE_BRANCH.replaceAll("[^a-zA-Z0-9]+", { r -> "-" })
                        sh "sed -i 's/${OLD_VERSION}/${VERSION}/' ./${env.APPLICATION_VERSION_FILE}"
                    }
                    stash includes: env.APPLICATION_VERSION_FILE, name: 'versionFile'
                    stash includes: '.bumpversion.cfg', name: 'bumpVersionFile'
                    currentBuild.displayName = "${env.APPLICATION_IMAGE}-v${VERSION}-${BUILD_NUMBER}"
                }
                echo """
********************
VERSION:\t${VERSION}
********************
"""
            }
        }
        stage('Build, Test and Publish') {
            agent {
                label "docker"
            }
            stages {
                stage('Build Docker Image') {
                    steps {
                        unstash 'versionFile'
                        script {
                            APPLICATION_IMAGE = docker.build(
                                env.APPLICATION_REGISTRY+"/${env.APPLICATION_IMAGE}:v${VERSION}","."
                            )                            
                        }
                    }
                }
                stage('Publish Image to ECR') {
                    when {
                        anyOf {
                            environment name: 'GIT_BRANCH', value: 'main'
                            environment name: 'GIT_BRANCH', value: 'origin/main'
                        }
                        expression {
                            params.RELEASE == true
                        }
                    }
                    steps {
                        script {
                            docker.withRegistry( "https://${env.APPLICATION_REGISTRY}", env.APPLICATION_REGISTRY_CREDENTIAL ) {
                                APPLICATION_IMAGE.push()
                            }
                            ARTIFACT_RELEASED = true
                        }
                    }                  
                }
                stage('Clean Docker Image') {
                    steps {
                        // always clean up image
                        sh "docker rmi ${env.APPLICATION_REGISTRY}/${env.APPLICATION_IMAGE}:v${VERSION}"
                    }                  
                }
            }
        }
        stage('Tag Git with Release'){
            agent {
                label "built-in"
            }
            when {
                expression {
                    ARTIFACT_RELEASED == true
                }
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'main'
                    environment name: 'GIT_BRANCH', value: 'origin/main'
                }
            }
            steps {
                unstash 'versionFile'
                unstash 'bumpVersionFile'
                sshagent([env.SSH_CREDENTIALS]) {
                    sh script: """
                        git add ${APPLICATION_VERSION_FILE} .bumpversion.cfg
                        git commit -m "bump ${params.VERSION_INCREMENT} version to ${VERSION}"
                        git push --set-upstream origin main
                    """
                }
                gitTagBranch version: "${VERSION}"
            }
        }
    }
    post {
        always {
            logstashSend failBuild: false, maxLines: 0
        }
        cleanup {
            cleanWs()
        }
    }
}