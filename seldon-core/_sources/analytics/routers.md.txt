# Seldon Core 中的路由

## 定义
路由器是 Seldon Core 中预定义类型的 [预测单元](../reference/apis/prediction.md#proto-buffer-and-grpc-definition)。
它是一种微服务，可将请求路由到其子节点之一，并可选择接收反馈奖励以做出路由选择。
REST 和 gRPC 内部 API，必须是路由器组件必须符合的 [内部 API](../reference/apis/internal-api.md#route)参考。

## 实现
目前我们在 Python 中提供了两个路由器的参考实现。他们都是 [multi-armed bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies) 的实现：
* [Epsilon-greedy router](https://github.com/SeldonIO/seldon-core/tree/master/components/routers/epsilon-greedy)
* [Thompson Sampling](https://github.com/SeldonIO/seldon-core/tree/master/components/routers/thompson-sampling)

## 实现自定义路由器
路由器组件必须实现一种 `Route` 方法，该方法将传入请求返回路由器组件的子路由之一。目前自定义路由器返回值的选项有

 * -1 : 路由到所有子路由
 * -2 : 不路由到任何子路由并返回当前请求作为响应
 * N >= 0 : 路由到子路由 N

REST 响应应为 SeldonMessage 负载返回，包含路由值或包含整数数组的 JSON。

可选地，`SendFeedback` 方法实现为提供一种机制来通知路由器其决策的质量。这将用于自适应路由器，例如 multi-armed bandits，有关更多详细信息，请参阅 [epsilon-greedy](https://github.com/SeldonIO/seldon-core/tree/master/components/routers/epsilon-greedy) 示例。

例如，考虑编写一个自定义 A/B/C... 测试组件，其中包含用户指定的子路由数和路由概率（Seldon Core 中已经支持双模型路由：[RANDOM_ABTEST](../reference/apis/prediction.md#proto-buffer-and-grpc-definition)）。
在这种情况下，因为路由逻辑是静态的，所以不需要实现 `SendFeedback`，因为我们不会通过为其路由选择提供反馈来动态更改路由。
另一方面，需要通过提供反馈来动态改变路由的自适应路由器将需要实现该 `SendFeedback` 方法。

由于路由器是只需要实现该 `Route` 方法的通用组件，因此在设计路由逻辑时具有相当大的灵活性。一些使用随机测试和 multi-armed bandits 的示例概念：
* 根据外部条件进行路由，例如使用一天中的时间将流量路由到已知在特定时间段内表现最佳的模型。
* 模型作为路由器：例如在路由器组件中使用预测模型首先确定更高级别的类成员（猫与狗），并根据决策将流量路由到更具体的模型（例如推断品种的狗特定模型）。

## 限制

当前 Go 中的默认编排「executor」在请求调用中未返回路由元数据。这是个 [已知问题](https://github.com/SeldonIO/seldon-core/issues/1823)。
