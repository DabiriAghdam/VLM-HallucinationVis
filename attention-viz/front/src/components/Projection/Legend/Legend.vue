<template>
    <Transition>
        <div class="legend-box" v-show="!renderState">
            <div id="legend" v-show="colorBy != 'no_outline'">
                <div class="bar-contain" :class="{
                        pos: colorBy == 'row' || colorBy == 'column' || colorBy == 'position' || colorBy == 'embed_norm' || colorBy == 'token_length' || colorBy == 'sent_length' || colorBy == 'token_freq', cat: colorBy == 'pos_mod_5', pun: colorBy == 'punctuation'
                    }">
                    <span>query</span>
                    <div class="bar">
                        <span class="low">{{ lowLabel() }}</span>
                        <span class="high">{{ highLabel() }}</span>
                    </div>
                </div>
                <div class="bar-contain k" :class="{
                    pos: colorBy == 'row' || colorBy == 'column' || colorBy == 'position' || colorBy == 'embed_norm' || colorBy == 'token_length' || colorBy == 'sent_length' || colorBy == 'token_freq', cat: colorBy == 'pos_mod_5', pun: colorBy == 'punctuation'
                }">
                    <span>key</span>
                    <div class="bar">
                        <span class="low">{{ lowLabel() }}</span>
                        <span class="high">{{ highLabel() }}</span>
                    </div>
                </div>
            </div>

            <p id="legend-msg" class="subtitle"><b>color info:</b> {{ colorMsg }}</p>
        </div>
    </Transition>
</template>

<script lang="ts">
import { onMounted, computed, reactive, toRefs, h, watch, ref, defineComponent } from "vue";
import { useStore } from "@/store/index";

export default defineComponent({
    setup(props, context) {
        const store = useStore();

        const state = reactive({
            colorBy: computed(() => store.state.colorBy),
            renderState: computed(() => store.state.renderState),
            modelType: computed(() => store.state.modelType),
            colorMsg: ""
        });

        const lowLabel = () => {
            switch (state.colorBy) {
                case 'pos_mod_5':
                case 'row':
                case 'column':
                    return "0"
                case 'position':
                    return "start"
                case 'token_length':
                case 'sent_length':
                    return "short"
                case 'embed_norm':
                case 'token_freq':
                    return "low"
                case 'punctuation':
                    return ".?!"
                default:
                    ""
            }
        }
        const highLabel = () => {
            switch (state.colorBy) {
                case 'position':
                    return "end"
                case 'row':
                case 'column':
                    return state.modelType == "vit-16" ? 13 : 6
                case 'pos_mod_5':
                    return "4"
                case 'embed_norm':
                case 'token_freq':
                    return "high"
                case 'token_length':
                case 'sent_length':
                    return "long"
                case 'punctuation':
                    return "abc"
                default:
                    ""
            }
        }

        // change msg below legend
        const setColorMsg = (msg: string) => {
            state.colorMsg = msg;
        }

        context.expose({
            setColorMsg
        });

        return {
            ...toRefs(state),
            highLabel,
            lowLabel
        };
    }
})

</script>

<style lang="scss">
.legend-box {
    transition: 0.5s;
    margin-top: 10px;
    width: 235px;
    display: flex;
    column-gap: 20px;
    align-items: center;
}

#legend {
    display: flex;
    column-gap: 12px;
}

#legend-msg {
    margin-bottom: 0;
}

.bar-contain {
    text-align: center;
    font-size: small;
}

/* default: type */
.bar {
    height: calc(24px + 0.2vw);
    max-height: 300px;
    width: calc(24px + 0.2vw);
    background: rgb(95, 185, 108);
    margin: 5px auto 0;
    transition: 0.5s;
    position: relative;
}

.bar-contain.k .bar {
    background: rgb(227, 55, 143);
}

/* position or norm */
.bar-contain.pos .bar {
    background: linear-gradient(to top, #ddefbb, #82CA7C, #00482A);
    height: calc(120px + 1vw);
}

.bar-contain.k.pos .bar {
    background: linear-gradient(to top, #ead4ed, #E33F97, #5E021B);
}

/* categorical */
.bar-contain.cat .bar {
    background: linear-gradient(#A144DB 20%,
            #528DDB 20% 40%,
            #5FB96C 40% 60%,
            #EDB50E 60% 80%,
            #E3378F 80%);
    height: calc(120px + 1vw);
}

.bar-contain.k.cat .bar {
    background: linear-gradient(#D6BAE3 20%,
            #C8DDED 20% 40%,
            #C4D6B8 40% 60%,
            #F0D6A5 60% 80%,
            #F5C0CA 80%);
}

/* categorical */
.bar-contain.pun .bar {
    background: linear-gradient(#F39226 50%, #5FB96C 50%);
    height: calc(48px + 0.4vw);
}

.bar-contain.k.pun .bar {
    background: linear-gradient(#E3378F 50%, #2E93D9 50%);
}

/* bar labels */
.bar span {
    display: block;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    font-size: x-small;
    transition: 0.5s;
    color: white;
}

.bar .high {
    top: 5px;
}

.bar .low {
    bottom: 5px;
}
</style>